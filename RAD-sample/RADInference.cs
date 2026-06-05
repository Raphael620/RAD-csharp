using MathNet.Numerics.LinearAlgebra;
using OpenCvSharp;
using OpenVinoSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Size = OpenCvSharp.Size;

public class RADInference : IDisposable
{
    private Scalar mean = new Scalar(0.485f, 0.456f, 0.406f);
    private Scalar std = new Scalar(0.229f, 0.224f, 0.225f);

    private Core core;
    private CompiledModel compiledModel;
    private InferRequest inferRequest;

    private int[] layers;                          // 层索引，例如 [3,6,9,11]
    private List<float[,]> clsBanks;               // 每层 CLS 特征 [N, C]
    private List<float[,,]> patchBanks;            // 每层 patch 特征 [N, L, C]
    private List<Matrix<float>> patchBanksMatrix;   // 预构建的 patch 矩阵 [N, L*C] ？实际用展平矩阵 [N*L, C] 但为了复用，我们存原始数据，运行时临时构造展平矩阵
    private List<float[,,]> patchBanksNorm;        // 预归一化的 patch 记忆库

    private int kImage;                             // 图像级检索数量
    private float[] layerWeights;                   // 层融合权重
    private int numLayers;
    private int embedDim;                            // 特征维度，例如 384
    private int numPatches;                           // patch 数量，784
    private int patchH, patchW;                        // 28x28

    // 常量
    private const int InputSize = 448;//448
    private const int ResizeSize = 512;//512

    //[DllImport("kernel32.dll")]
    //private static extern bool AllocConsole();

    //[DllImport("kernel32.dll")]
    //private static extern bool FreeConsole();

    public RADInference(string onnxPath, string bankDir, int kImage = 150, float[] layerWeights = null)
    {
        this.kImage = kImage;
        //AllocConsole();
        // 1. 加载 ONNX 模型
        core = new Core();
        var devices=core.get_available_devices();
        Debug.WriteLine(string.Join(",",devices));
        string selectedDevice=devices.Contains("GPU.0")?"GPU.0":"CPU";
        var model = core.read_model(onnxPath);
        compiledModel = core.compile_model(model, selectedDevice);
        inferRequest = compiledModel.create_infer_request();

        // 2. 加载记忆库
        var layersPath = Path.Combine(bankDir, "layers.bin");
        layers = BinaryArrayReader.ReadLayers(layersPath);
        numLayers = layers.Length;

        clsBanks = new List<float[,]>();
        patchBanks = new List<float[,,]>();
        for (int i = 0; i < numLayers; i++)
        {
            int layer = layers[i];
            var cls = BinaryArrayReader.Read2DArray(Path.Combine(bankDir, $"cls_layer{layer}.bin"));
            var patch = BinaryArrayReader.Read3DArray(Path.Combine(bankDir, $"patch_layer{layer}.bin"));
            clsBanks.Add(cls);
            patchBanks.Add(patch);
        }

        // 3. 预归一化 patch 记忆库
        patchBanksNorm = new List<float[,,]>();
        foreach (var pb in patchBanks)
        {
            int n = pb.GetLength(0);
            int l = pb.GetLength(1);
            int c = pb.GetLength(2);
            float[,,] pbNorm = new float[n, l, c];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < l; j++)
                {
                    float norm = 0f;
                    for (int k = 0; k < c; k++)
                        norm += pb[i, j, k] * pb[i, j, k];
                    norm = (float)Math.Sqrt(norm + 1e-8f);
                    for (int k = 0; k < c; k++)
                        pbNorm[i, j, k] = pb[i, j, k] / norm;
                }
            patchBanksNorm.Add(pbNorm);
        }

        // 4. 层权重
        if (layerWeights == null || layerWeights.Length != numLayers)
        {
            this.layerWeights = Enumerable.Repeat(1f / numLayers, numLayers).ToArray();
        }
        else
        {
            float sum = layerWeights.Sum();
            this.layerWeights = layerWeights.Select(w => w / sum).ToArray();
        }

        // 5. 维度信息
        embedDim = clsBanks[0].GetLength(1);
        numPatches = patchBanks[0].GetLength(1);
        patchH = patchW = (int)Math.Sqrt(numPatches);
    }

    /// <summary>预处理图像，返回 NCHW 顺序的 float[]</summary>
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

        Cv2.Subtract(cropped, mean, cropped);
        Cv2.Divide(cropped, std, cropped);

        float[] inputData = RADTool.common.ExtractMat(cropped);
        return inputData;
    }

    /// <summary>获取指定输出张量的数据</summary>
    private float[] GetOutputData(string name)
    {
        var tensor = inferRequest.get_tensor(name);
        return tensor.get_data<float>((int)tensor.size);
    }

    /// <summary>双线性插值上采样</summary>
    private float[,] ResizeBilinear(float[,] src, int dstH, int dstW)
    {
        int srcH = src.GetLength(0);
        int srcW = src.GetLength(1);
        float[,] dst = new float[dstH, dstW];
        float scaleX = (srcW - 1f) / (dstW - 1f);
        float scaleY = (srcH - 1f) / (dstH - 1f);

        for (int y = 0; y < dstH; y++)
        {
            float fy = y * scaleY;
            int y0 = (int)fy;
            int y1 = Math.Min(y0 + 1, srcH - 1);
            float wy1 = fy - y0;
            float wy0 = 1 - wy1;

            for (int x = 0; x < dstW; x++)
            {
                float fx = x * scaleX;
                int x0 = (int)fx;
                int x1 = Math.Min(x0 + 1, srcW - 1);
                float wx1 = fx - x0;
                float wx0 = 1 - wx1;

                float v00 = src[y0, x0];
                float v01 = src[y0, x1];
                float v10 = src[y1, x0];
                float v11 = src[y1, x1];

                float v0 = v00 * wx0 + v01 * wx1;
                float v1 = v10 * wx0 + v11 * wx1;
                dst[y, x] = v0 * wy0 + v1 * wy1;
            }
        }
        return dst;
    }

    /// <summary>生成高斯核</summary>
    private float[,] GaussianKernel(int size = 5, float sigma = 1.0f)
    {
        float[,] kernel = new float[size, size];
        float sum = 0;
        int half = size / 2;
        for (int y = -half; y <= half; y++)
        {
            for (int x = -half; x <= half; x++)
            {
                float val = (float)Math.Exp(-(x * x + y * y) / (2 * sigma * sigma));
                kernel[y + half, x + half] = val;
                sum += val;
            }
        }
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                kernel[i, j] /= sum;
        return kernel;
    }

    /// <summary>对二维数组进行高斯卷积</summary>
    private float[,] Convolve2D(float[,] src, float[,] kernel)
    {
        int h = src.GetLength(0);
        int w = src.GetLength(1);
        int kh = kernel.GetLength(0);
        int kw = kernel.GetLength(1);
        int padH = kh / 2;
        int padW = kw / 2;

        float[,] dst = new float[h, w];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float sum = 0;
                for (int ky = 0; ky < kh; ky++)
                {
                    int sy = y + ky - padH;
                    if (sy < 0 || sy >= h) continue;
                    for (int kx = 0; kx < kw; kx++)
                    {
                        int sx = x + kx - padW;
                        if (sx < 0 || sx >= w) continue;
                        sum += src[sy, sx] * kernel[ky, kx];
                    }
                }
                dst[y, x] = sum;
            }
        }
        return dst;
    }

    /// <summary>执行推理，返回异常图和图像级分数（优化版）</summary>
    public (float[,] anomalyMap, float imageScore) Infer(string imagePath, int outputSize = 256, bool applySmoothing = true, Action<string> progressCallback = null)
    {
        progressCallback?.Invoke("预处理图像...");
        float[] inputData = PreprocessImage(imagePath);
        var inputTensor = new Tensor(new Shape([1, 3, InputSize, InputSize]), inputData);
        inferRequest.set_input_tensor(0, inputTensor);

        progressCallback?.Invoke("推理中...");
        Stopwatch stopwatch = Stopwatch.StartNew();
        inferRequest.infer();

        stopwatch.Stop();
        Console.WriteLine($"{stopwatch.ElapsedMilliseconds}ms");


        progressCallback?.Invoke("获取输出...");
        List<float[,,]> patchList = new List<float[,,]>();
        List<float[,]> clsList = new List<float[,]>();
        foreach (int layer in layers)
        {
            string patchName = $"patch_{layer}";
            string clsName = $"cls_{layer}";
            float[] patchFlat = GetOutputData(patchName);
            float[] clsFlat = GetOutputData(clsName);

            float[,,] patch3D = new float[1, numPatches, embedDim];
            Buffer.BlockCopy(patchFlat, 0, patch3D, 0, patchFlat.Length * sizeof(float));
            patchList.Add(patch3D);

            float[,] cls2D = new float[1, embedDim];
            Buffer.BlockCopy(clsFlat, 0, cls2D, 0, clsFlat.Length * sizeof(float));
            clsList.Add(cls2D);
        }

        progressCallback?.Invoke("图像级检索...");
        float[] queryCls = GetRow(clsList.Last(), 0);
        float[] normQueryCls = Normalize(queryCls);
        float[,] clsBankGlobal = clsBanks.Last();
        int N = clsBankGlobal.GetLength(0);
        float[] similarities = new float[N];
        for (int i = 0; i < N; i++)
        {
            float[] trainCls = GetRow(clsBankGlobal, i);
            float normTrain = (float)Math.Sqrt(Dot(trainCls, trainCls) + 1e-8);
            similarities[i] = Dot(trainCls, normQueryCls) / normTrain;
        }
        int[] topkIndices = GetTopKIndices(similarities, kImage);

        // 逐层 patch-KNN（优化版本：矩阵乘法 + 并行）
        float[] fusedScores = new float[numPatches];
        object lockObj = new object();

        Parallel.For(0, numLayers, li =>
        {
            progressCallback?.Invoke($"处理层 {layers[li]}...");

            // 获取测试 patch 特征并归一化
            float[,,] testPatch = patchList[li];
            float[,] testPatch2D = new float[numPatches, embedDim];
            for (int p = 0; p < numPatches; p++)
                for (int c = 0; c < embedDim; c++)
                    testPatch2D[p, c] = testPatch[0, p, c];

            // 归一化测试 patch (逐行归一化)
            var testNormMatrix = Matrix<float>.Build.Dense(numPatches, embedDim);
            for (int p = 0; p < numPatches; p++)
            {
                float norm = 0f;
                for (int c = 0; c < embedDim; c++)
                    norm += testPatch2D[p, c] * testPatch2D[p, c];
                norm = (float)Math.Sqrt(norm + 1e-8f);
                for (int c = 0; c < embedDim; c++)
                    testNormMatrix.At(p, c, testPatch2D[p, c] / norm);
            }

            // 构建 bank 矩阵 [k * L, C]（从预归一化的 bank 中提取）
            float[,,] bankNorm = patchBanksNorm[li];
            int k = topkIndices.Length;
            var bankMatrix = Matrix<float>.Build.Dense(k * numPatches, embedDim);
            for (int idx = 0; idx < k; idx++)
            {
                int sampleIdx = topkIndices[idx];
                for (int p = 0; p < numPatches; p++)
                {
                    int row = idx * numPatches + p;
                    for (int c = 0; c < embedDim; c++)
                        bankMatrix.At(row, c, bankNorm[sampleIdx, p, c]);
                }
            }

            // 矩阵乘法：similarity = testNormMatrix * bankMatrix^T   [L, k*L]
            var similarity = testNormMatrix * bankMatrix.Transpose();

            // 对每一行取最大值，得到异常分数
            float[] layerScores = new float[numPatches];
            for (int p = 0; p < numPatches; p++)
            {
                float maxSim = -1f;
                for (int j = 0; j < similarity.ColumnCount; j++)
                {
                    float sim = similarity.At(p, j);
                    if (sim > maxSim) maxSim = sim;
                }
                layerScores[p] = 1f - maxSim;
            }

            // 加权融合（线程安全）
            lock (lockObj)
            {
                for (int p = 0; p < numPatches; p++)
                    fusedScores[p] += layerWeights[li] * layerScores[p];
            }
        });

        progressCallback?.Invoke("生成异常图...");
        float[,] patchMap = new float[patchH, patchW];
        for (int i = 0; i < patchH; i++)
            for (int j = 0; j < patchW; j++)
                patchMap[i, j] = fusedScores[i * patchW + j];

        float[,] anomalyMap = ResizeBilinear(patchMap, outputSize, outputSize);

        if (applySmoothing)
        {
            progressCallback?.Invoke("高斯平滑...");
            var kernel = GaussianKernel(5, 1.0f);
            anomalyMap = Convolve2D(anomalyMap, kernel);
        }

        float imageScore = 0;
        for (int i = 0; i < outputSize; i++)
            for (int j = 0; j < outputSize; j++)
                if (anomalyMap[i, j] > imageScore)
                    imageScore = anomalyMap[i, j];

        progressCallback?.Invoke("完成");
        return (anomalyMap, imageScore);
    }

    // ---------- 辅助函数 ----------
    private float[] GetRow(float[,] arr, int row)
    {
        int cols = arr.GetLength(1);
        float[] result = new float[cols];
        for (int j = 0; j < cols; j++)
            result[j] = arr[row, j];
        return result;
    }

    private float Dot(float[] a, float[] b)
    {
        float sum = 0;
        for (int i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return sum;
    }

    private float[] Normalize(float[] vec)
    {
        float norm = (float)Math.Sqrt(Dot(vec, vec) + 1e-8f);
        float[] result = new float[vec.Length];
        for (int i = 0; i < vec.Length; i++)
            result[i] = vec[i] / norm;
        return result;
    }

    private int[] GetTopKIndices(float[] arr, int k)
    {
        return arr.Select((val, idx) => new { val, idx })
                  .OrderByDescending(x => x.val)
                  .Take(k)
                  .Select(x => x.idx)
                  .ToArray();
    }

    public void Dispose()
    {
        inferRequest?.Dispose();
        compiledModel?.Dispose();
        core?.Dispose();
    }
}