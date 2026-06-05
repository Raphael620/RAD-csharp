using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using OpenCvSharp;
using OpenVinoSharp;
using RADTool;
using Size = OpenCvSharp.Size;

public class MemoryBankBuilder : IDisposable
{
    private Core core;
    private CompiledModel compiledModel;
    private InferRequest inferRequest;

    private readonly int[] layers = { 3, 6, 9, 11 };
    private readonly int embedDim = 384;      // ViT-S/16
    private readonly int numPatches = 784;

    private const int InputSize = 448;//448
    private const int ResizeSize = 512;//512

    public MemoryBankBuilder(string onnxPath, string device = "CPU")
    {
        core = new Core();
        var model = core.read_model(onnxPath);
        compiledModel = core.compile_model(model, device);
        inferRequest = compiledModel.create_infer_request();
    }

    public async Task BuildAsync(string imageFolder, string outputDir, IProgress<string> progress)
    {
        // 获取所有图像
        var imagePaths = Directory.GetFiles(imageFolder, "*.*", SearchOption.AllDirectories)
            .Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                        f.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ||
                        f.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase))
            .ToArray();

        if (imagePaths.Length == 0)
            throw new Exception("未找到图像文件");

        progress.Report($"找到 {imagePaths.Length} 张图像，开始提取特征...");

        // 特征收集
        List<float[,]> clsList = new List<float[,]>();
        List<float[,,]> patchList = new List<float[,,]>();
        for (int i = 0; i < layers.Length; i++)
        {
            clsList.Add(new float[imagePaths.Length, embedDim]);
            patchList.Add(new float[imagePaths.Length, numPatches, embedDim]);
        }

        int batchSize = 16;
        for (int start = 0; start < imagePaths.Length; start += batchSize)
        {
            int end = Math.Min(start + batchSize, imagePaths.Length);
            int batchCount = end - start;

            var batchData = new List<float[]>();
            for (int idx = start; idx < end; idx++)
            {
                batchData.Add(PreprocessImage(imagePaths[idx]));
            }

            // 构建输入张量（修正语法）
            var inputTensor = new Tensor(new Shape([batchCount, 3, InputSize, InputSize]), ElementType.F32);
            unsafe
            {
                float* ptr = (float*)inputTensor.data();
                foreach (var data in batchData)
                {
                    Marshal.Copy(data, 0, (IntPtr)ptr, data.Length);
                    ptr += data.Length;
                }
            }

            inferRequest.set_input_tensor(0, inputTensor);
            inferRequest.infer();

            for (int li = 0; li < layers.Length; li++)
            {
                int layer = layers[li];
                string patchName = $"patch_{layer}";
                string clsName = $"cls_{layer}";

                Debug.WriteLine($"处理层 {layer}，批大小 {batchCount}");

                var patchTensor = inferRequest.get_tensor(patchName);
                var clsTensor = inferRequest.get_tensor(clsName);

                int expectedPatchLen = batchCount * numPatches * embedDim;
                int expectedClsLen = batchCount * embedDim;

                float[] patchData = patchTensor.get_data<float>(expectedPatchLen);
                float[] clsData = clsTensor.get_data<float>(expectedClsLen);

                Debug.WriteLine($"patchData.Length = {patchData.Length}, expected = {expectedPatchLen}");
                Debug.WriteLine($"clsData.Length = {clsData.Length}, expected = {expectedClsLen}");

                if (patchData.Length != expectedPatchLen)
                    throw new Exception($"层 {layer} patch 长度不一致：实际 {patchData.Length}，预期 {expectedPatchLen}。请检查 embedDim 或 numPatches 设置，或模型输出名称是否正确。");
                if (clsData.Length != expectedClsLen)
                    throw new Exception($"层 {layer} cls 长度不一致：实际 {clsData.Length}，预期 {expectedClsLen}。请检查 embedDim 设置。");

                // 使用 Buffer.BlockCopy 替代 Array.Copy，避免多维数组转换问题
                int clsSize = embedDim * sizeof(float);
                int patchSize = embedDim * sizeof(float);

                for (int b = 0; b < batchCount; b++)
                {
                    // CLS 复制
                    int clsSrcOffset = b * embedDim;
                    int clsDstOffset = (start + b) * embedDim;
                    try
                    {
                        Buffer.BlockCopy(clsData, clsSrcOffset * sizeof(float), clsList[li], clsDstOffset * sizeof(float), clsSize);
                    }
                    catch (Exception ex)
                    {
                        throw new Exception($"复制 cls 数据失败: {ex.Message}，源偏移 {clsSrcOffset}，目标偏移 {clsDstOffset}，长度 {embedDim}。clsList[{li}] 形状 = [{clsList[li].GetLength(0)}, {clsList[li].GetLength(1)}]，当前批次 start={start}, b={b}, batchCount={batchCount}");
                    }

                    // Patch 复制
                    int patchSrcOffset = b * numPatches * embedDim;
                    int patchDstBase = (start + b) * numPatches * embedDim;
                    for (int p = 0; p < numPatches; p++)
                    {
                        int srcPos = patchSrcOffset + p * embedDim;
                        int dstPos = patchDstBase + p * embedDim;
                        try
                        {
                            Buffer.BlockCopy(patchData, srcPos * sizeof(float), patchList[li], dstPos * sizeof(float), patchSize);
                        }
                        catch (Exception ex)
                        {
                            throw new Exception($"复制 patch 数据失败: {ex.Message}，层 {layer}，patch {p}，源偏移 {srcPos}，目标偏移 {dstPos}，长度 {embedDim}");
                        }
                    }
                }
            }

            progress.Report($"已处理 {end}/{imagePaths.Length} 张图像");
        }

        // 保存到文件
        Directory.CreateDirectory(outputDir);
        BinaryArrayHelper.SaveLayers(layers, Path.Combine(outputDir, "layers.bin"));
        for (int li = 0; li < layers.Length; li++)
        {
            BinaryArrayHelper.Save2DArray(clsList[li], Path.Combine(outputDir, $"cls_layer{layers[li]}.bin"));
            BinaryArrayHelper.Save3DArray(patchList[li], Path.Combine(outputDir, $"patch_layer{layers[li]}.bin"));
        }

        progress.Report("记忆库构建完成！");
    }

    private float[] PreprocessImage(string imagePath)
    {
        using var src = Cv2.ImRead(imagePath, ImreadModes.Color);
        if (src.Empty()) throw new Exception("无法读取图像");

        using var resized = new Mat();
        Cv2.Resize(src, resized, new Size(ResizeSize, ResizeSize));

        int cropSize = InputSize;
        int startX = (ResizeSize - cropSize) / 2;
        int startY = (ResizeSize - cropSize) / 2;
        using var cropped = new Mat(resized, new OpenCvSharp.Rect(startX, startY, cropSize, cropSize));

        cropped.ConvertTo(cropped, MatType.CV_32FC3, 1.0 / 255);
        Cv2.CvtColor(cropped, cropped, ColorConversionCodes.BGR2RGB);

        Scalar mean = new Scalar(0.485f, 0.456f, 0.406f);
        Scalar std = new Scalar(0.229f, 0.224f, 0.225f);
        Cv2.Subtract(cropped, mean, cropped);
        Cv2.Divide(cropped, std, cropped);
        float[] inputData = common.ExtractMat(cropped);
        return inputData;
    }

    public void Dispose()
    {
        inferRequest?.Dispose();
        compiledModel?.Dispose();
        core?.Dispose();
    }
}