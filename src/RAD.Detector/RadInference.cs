using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;

namespace RAD.Detector;

/// <summary>
/// Core RAD inference engine using ONNX Runtime.
/// Supports building memory bank from normal samples and detecting anomalies.
/// </summary>
public sealed class RadInference : IDisposable
{
    private Microsoft.ML.OnnxRuntime.InferenceSession _session;
    private readonly OnnxModelInfo _modelInfo;

    // Bank data
    private int[]? _layers;
    private float[][]? _clsBanks;       // [numLayers][N * C]
    private float[][][]? _patchBanks;   // [numLayers][N][L*C]
    private float[][][]? _patchBanksNorm; // pre-normalized patch banks

    private float[] _layerWeights;
    private int _kImage = 150;

    public bool IsBankLoaded => _clsBanks != null;

    // kImage can be changed at runtime without rebuilding bank
    public int KImage { get => _kImage; set => _kImage = Math.Max(1, value); }

    public string ActiveDevice { get; private set; }
    private readonly string _onnxPath;

    public RadInference(string onnxPath, string device = "CPU", int kImage = 10)
    {
        _onnxPath = onnxPath;
        _kImage = kImage;
        CreateSession(device);
        _modelInfo = OnnxModelInfo.FromSession(_session);
        _layerWeights = Enumerable.Repeat(1f / _modelInfo.LayerIndices.Length, _modelInfo.LayerIndices.Length).ToArray();
    }

    public string SwitchDevice(string device = "CPU")
    {
        _session?.Dispose();
        CreateSession(device);
        return ActiveDevice;
    }

    private void CreateSession(string device)
    {
        var options = new Microsoft.ML.OnnxRuntime.SessionOptions();

        bool useGpu = device.Equals("GPU", StringComparison.OrdinalIgnoreCase) ||
                      device.Equals("DML", StringComparison.OrdinalIgnoreCase);

        if (useGpu)
        {
            try
            {
                options.AppendExecutionProvider_DML(0);
                ActiveDevice = "GPU (DirectML)";
            }
            catch
            {
                options.AppendExecutionProvider_CPU();
                ActiveDevice = "CPU (DML unavailable)";
            }
        }
        else
        {
            options.AppendExecutionProvider_CPU();
            ActiveDevice = "CPU";
        }

        _session = new Microsoft.ML.OnnxRuntime.InferenceSession(_onnxPath, options);
    }

    /// <summary>
    /// Build memory bank from a directory of normal images. Synchronous.
    /// Collects all features first, then generates bank.
    /// </summary>
    public (int totalProcessed, int totalFiles) BuildBank(string normalDir, string bankOutputDir, Action<string>? onProgress = null)
    {
        var imageFiles = Directory.EnumerateFiles(normalDir)
            .Where(f => f.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ||
                        f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                        f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
                        f.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase))
            .OrderBy(f => f)
            .ToArray();

        if (imageFiles.Length == 0)
            throw new InvalidOperationException($"No images found in {normalDir}");

        int numLayers = _modelInfo.LayerIndices.Length;
        int embedDim = _modelInfo.EmbedDim;
        int numPatches = _modelInfo.NumPatches;

        // --- Phase 1: process each image, collect features ---
        var clsFeats = new List<float[]>[numLayers];
        var patchFeats = new List<float[]>[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            clsFeats[i] = new List<float[]>();
            patchFeats[i] = new List<float[]>();
        }

        for (int i = 0; i < imageFiles.Length; i++)
        {
            var file = imageFiles[i];
            onProgress?.Invoke($"Processing [{i + 1}/{imageFiles.Length}]: {Path.GetFileName(file)}");

            try
            {
                float[] input = ImagePreprocessor.Preprocess(file);
                float[][] patchOuts, clsOuts;
                RunInference(input, out patchOuts, out clsOuts, out _);

                for (int li = 0; li < numLayers; li++)
                {
                    clsFeats[li].Add(clsOuts[li]);
                    patchFeats[li].Add(patchOuts[li]);
                }
            }
            catch (Exception ex)
            {
                onProgress?.Invoke($"Skipping {Path.GetFileName(file)}: {ex.Message}");
            }
        }

        int totalProcessed = clsFeats[0].Count;
        if (totalProcessed == 0)
            throw new InvalidOperationException("No images could be processed. Check image formats.");

        // --- Phase 2: build bank arrays ---
        _layers = _modelInfo.LayerIndices;
        _clsBanks = new float[numLayers][];
        _patchBanks = new float[numLayers][][];
        _patchBanksNorm = new float[numLayers][][];

        for (int li = 0; li < numLayers; li++)
        {
            int N = clsFeats[li].Count;
            _clsBanks[li] = new float[N * embedDim];
            _patchBanks[li] = new float[N][];
            _patchBanksNorm[li] = new float[N][];

            for (int n = 0; n < N; n++)
            {
                Array.Copy(clsFeats[li][n], 0, _clsBanks[li], n * embedDim, embedDim);
                _patchBanks[li][n] = patchFeats[li][n];
                _patchBanksNorm[li][n] = NormalizePerPatch(patchFeats[li][n], embedDim);
            }
        }

        // --- Phase 3: save to bin ---
        BankSerializer.SaveBank(bankOutputDir, _layers, _clsBanks, _patchBanks);

        onProgress?.Invoke($"Bank built with {totalProcessed}/{imageFiles.Length} images, saved to {bankOutputDir}");
        return (totalProcessed, imageFiles.Length);
    }

    /// <summary>
    /// Load a pre-built bank from directory.
    /// </summary>
    public void LoadBank(string bankDir)
    {
        var (layers, clsBanks, patchBanks) = BankSerializer.LoadBank(bankDir);
        _layers = layers;
        _clsBanks = clsBanks;

        _patchBanks = patchBanks;
        _patchBanksNorm = new float[layers.Length][][];
        for (int li = 0; li < layers.Length; li++)
        {
            _patchBanksNorm[li] = new float[patchBanks[li].Length][];
            for (int n = 0; n < patchBanks[li].Length; n++)
                _patchBanksNorm[li][n] = NormalizePerPatch(patchBanks[li][n], _modelInfo.EmbedDim);
        }
    }

    /// <summary>
    public (float[,] anomalyMap, float score) Detect(
        string imagePath, float threshold,
        out double preMs, out double boneMs, out double postMs,
        int outputSize = 256, bool applySmoothing = true)
    {
        if (_clsBanks == null || _patchBanks == null || _layers == null)
            throw new InvalidOperationException("Bank not loaded. Build or load a bank first.");

        preMs = boneMs = postMs = 0;
        var sw = System.Diagnostics.Stopwatch.StartNew();

        float[] input = ImagePreprocessor.Preprocess(imagePath);
        preMs = sw.Elapsed.TotalMilliseconds;
        sw.Restart();

        float[][] patchOuts, clsOuts;
        RunInference(input, out patchOuts, out clsOuts, out _);
        boneMs = sw.Elapsed.TotalMilliseconds;
        sw.Restart();

        var (anomalyMap, score) = ComputeAnomalyMap1(patchOuts, clsOuts, outputSize, applySmoothing);
        postMs = sw.Elapsed.TotalMilliseconds;

        return (anomalyMap, score);
    }

    
    /// <summary>
    /// Run ONNX inference. Returns patch and cls outputs per layer + actual C dimension.
    /// </summary>
    private void RunInference(float[] input, out float[][] patchOuts, out float[][] clsOuts, out int actualC)
    {
        actualC = 0;
        int[] dims = [1, 3, 448, 448];
        var denseTensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(dims);
        input.AsSpan().CopyTo(denseTensor.Buffer.Span);
        var namedInput = NamedOnnxValue.CreateFromTensor("input", denseTensor);

        var inputs = new List<NamedOnnxValue> { namedInput };

        using var results = _session.Run(inputs);

        var dict = new Dictionary<string, float[]>();
        foreach (var result in results)
        {
            var t = result.AsTensor<float>();
            dict[result.Name] = t.ToArray();
            if (result.Name == $"patch_{_modelInfo.LayerIndices[0]}")
                actualC = (int)t.Dimensions[2];
        }

        int numLayers = _modelInfo.LayerIndices.Length;
        patchOuts = new float[numLayers][];
        clsOuts = new float[numLayers][];
        for (int li = 0; li < numLayers; li++)
        {
            patchOuts[li] = dict[$"patch_{_modelInfo.LayerIndices[li]}"];
            clsOuts[li] = dict[$"cls_{_modelInfo.LayerIndices[li]}"];
        }
    }

    /*
    private (float[,] anomalyMap, float score) ComputeAnomalyMap(
        float[][] queryPatch, float[][] queryCls, int outputSize, bool applySmoothing)
    {
        int numLayers = _layers!.Length;
        int numPatches = _modelInfo.NumPatches;
        int embedDim = _modelInfo.EmbedDim;
        int N = _clsBanks![0].Length / embedDim;

        // === Step 5: Image-level retrieval (Python: cls_banks[-1] vs query_cls) ===
        float[] queryClsGlobal = Normalize(queryCls[numLayers - 1]);
        float[] clsBankGlobal = _clsBanks[numLayers - 1];

        var clsNorms = new float[N];
        for (int n = 0; n < N; n++)
        {
            float sum = 0;
            int off = n * embedDim;
            for (int c = 0; c < embedDim; c++)
                sum += clsBankGlobal[off + c] * clsBankGlobal[off + c];
            clsNorms[n] = MathF.Sqrt(sum) + 1e-8f;
        }

        var similarities = new float[N];
        for (int n = 0; n < N; n++)
        {
            float dot = 0;
            int off = n * embedDim;
            for (int c = 0; c < embedDim; c++)
                dot += queryClsGlobal[c] * clsBankGlobal[off + c];
            similarities[n] = dot / clsNorms[n];
        }

        int k = Math.Min(_kImage, N);
        var sortedIdx = Enumerable.Range(0, N).OrderByDescending(i => similarities[i]).Take(k).ToArray();

        // === Step 6: Multi-layer patch KNN (MathNet matrix multiply: [L,C] @ [C,k*L]^T = [L,k*L]) ===
        float[] fusedScores = new float[numPatches];
        object lockObj = new object();

        Parallel.For(0, numLayers, li =>
        {
            float[] qPatchNorm = NormalizePerPatch(queryPatch[li], embedDim);

            // Build query matrix [L, C]
            var qMat = MathNet.Numerics.LinearAlgebra.Matrix<float>.Build.Dense(numPatches, embedDim);
            for (int p = 0; p < numPatches; p++)
            {
                int off = p * embedDim;
                for (int c = 0; c < embedDim; c++)
                    qMat.At(p, c, qPatchNorm[off + c]);
            }

            // Build bank matrix [k*L, C]
            int kL = k * numPatches;
            var bMat = MathNet.Numerics.LinearAlgebra.Matrix<float>.Build.Dense(kL, embedDim);
            for (int ki = 0; ki < k; ki++)
            {
                float[] src = _patchBanksNorm![li][sortedIdx[ki]];
                for (int p = 0; p < numPatches; p++)
                {
                    int row = ki * numPatches + p;
                    int off = p * embedDim;
                    for (int c = 0; c < embedDim; c++)
                        bMat.At(row, c, src[off + c]);
                }
            }

            // simMatrix = qMat * bMat^T  =>  [L, k*L]
            var simMat = qMat * bMat.Transpose();

            float[] layerScores = new float[numPatches];
            for (int p = 0; p < numPatches; p++)
            {
                float maxSim = -1f;
                for (int bi = 0; bi < kL; bi++)
                {
                    float sim = simMat.At(p, bi);
                    if (sim > maxSim) maxSim = sim;
                }
                layerScores[p] = 1f - maxSim;
            }

            lock (lockObj)
            {
                for (int p = 0; p < numPatches; p++)
                    fusedScores[p] += _layerWeights[li] * layerScores[p];
            }
        });

        // === Step 7-9: reshape, bilinear upsample, gaussian ===
        int pH = _modelInfo.PatchH, pW = _modelInfo.PatchW;
        float[,] patchMap = new float[pH, pW];
        for (int y = 0; y < pH; y++)
            for (int x = 0; x < pW; x++)
                patchMap[y, x] = fusedScores[y * pW + x];

        float[,] anomalyMap = BilinearResize(patchMap, outputSize, outputSize);
        if (applySmoothing)
            anomalyMap = GaussianBlur(anomalyMap, 5, 1.0f);

        float score = 0;
        foreach (var v in anomalyMap)
            if (v > score) score = v;

        return (anomalyMap, score);
    }
    */
    private (float[,] anomalyMap, float score) ComputeAnomalyMap1(
    float[][] queryPatch, float[][] queryCls, int outputSize, bool applySmoothing)
    {
        int numLayers = _layers!.Length;
        int numPatches = _modelInfo.NumPatches;
        int embedDim = _modelInfo.EmbedDim;
        int N = _clsBanks![0].Length / embedDim;

        // ----- 一次性归一化所有层的 query patches（留在数组里）-----
        float[][] queryPatchNorm = new float[numLayers][];
        for (int li = 0; li < numLayers; li++)
            queryPatchNorm[li] = NormalizePerPatch(queryPatch[li], embedDim);

        // ----- Step 5: 图像级检索 -----
        float[] queryClsGlobal = Normalize(queryCls[numLayers - 1]);
        float[] clsBankGlobal = _clsBanks[numLayers - 1];

        // 计算与 N 个库向量的余弦相似度（因为库向量已归一化，直接用点积）
        var similarities = new float[N];
        var qSpan = new ReadOnlySpan<float>(queryClsGlobal);
        for (int n = 0; n < N; n++)
            similarities[n] = TensorPrimitives.Dot(qSpan,
                clsBankGlobal.AsSpan(n * embedDim, embedDim));

        // 取 TopK（最小堆）
        int k = Math.Min(_kImage, N);
        int[] sortedIdx = TopKIndices(similarities, k);

        // ----- Step 6: 多层 patch KNN -----
        float[][] layerScoresAll = new float[numLayers][];
        for (int li = 0; li < numLayers; li++)
            layerScoresAll[li] = new float[numPatches];

        Parallel.For(0, numLayers, li =>
        {
            float[] qNorm = queryPatchNorm[li];
            float[] layerScores = layerScoresAll[li];
            for (int p = 0; p < numPatches; p++)
            {
                var qPatchSpan = qNorm.AsSpan(p * embedDim, embedDim);
                float maxSim = -1f;
                for (int ki = 0; ki < k; ki++)
                {
                    float[] bankPatch = _patchBanksNorm![li][sortedIdx[ki]];
                    var bPatchSpan = bankPatch.AsSpan(p * embedDim, embedDim);
                    float sim = TensorPrimitives.Dot(qPatchSpan, bPatchSpan);
                    if (sim > maxSim) maxSim = sim;
                }
                layerScores[p] = 1f - maxSim;
            }
        });

        // 加权融合（无锁）
        float[] fusedScores = new float[numPatches];
        for (int li = 0; li < numLayers; li++)
        {
            float w = _layerWeights[li];
            float[] layer = layerScoresAll[li];
            for (int p = 0; p < numPatches; p++)
                fusedScores[p] += w * layer[p];
        }

        // ----- Step 7‑9: 重排与上采样 -----
        int pH = _modelInfo.PatchH, pW = _modelInfo.PatchW;
        float[,] patchMap = new float[pH, pW];
        Buffer.BlockCopy(fusedScores, 0, patchMap, 0, fusedScores.Length * sizeof(float));
        // （注意：BlockCopy 要求行优先布局匹配）

        float[,] anomalyMap = BilinearResize(patchMap, outputSize, outputSize);
        if (applySmoothing)
            anomalyMap = GaussianBlur(anomalyMap, 5, 1.0f);

        // 取全局最大值
        float score = anomalyMap.Cast<float>().Max();

        return (anomalyMap, score);
    }


    #region Math helpers

    /// <summary>
    /// Normalize each patch (C-dim segment) independently, matching Python:
    /// norms = np.linalg.norm(pb, axis=2, keepdims=True); pb_norm = pb / norms
    /// </summary>
    private static float[] NormalizePerPatch0(float[] flat, int embedDim)
    {
        int numPatches = flat.Length / embedDim;
        float[] result = new float[flat.Length];
        for (int p = 0; p < numPatches; p++)
        {
            int off = p * embedDim;
            float norm = 0;
            for (int c = 0; c < embedDim; c++)
                norm += flat[off + c] * flat[off + c];
            norm = MathF.Sqrt(norm) + 1e-8f;
            for (int c = 0; c < embedDim; c++)
                result[off + c] = flat[off + c] / norm;
        }
        return result;
    }

    private static float[] Normalize(float[] vec)
    {
        float norm = 0;
        foreach (var v in vec) norm += v * v;
        norm = MathF.Sqrt(norm) + 1e-8f;
        float[] result = new float[vec.Length];
        for (int i = 0; i < vec.Length; i++)
            result[i] = vec[i] / norm;
        return result;
    }

    private static float[,] BilinearResize(float[,] src, int dstH, int dstW)
    {
        int srcH = src.GetLength(0), srcW = src.GetLength(1);
        float[,] dst = new float[dstH, dstW];

        float scaleX = (float)(srcW - 1) / Math.Max(dstW - 1, 1);
        float scaleY = (float)(srcH - 1) / Math.Max(dstH - 1, 1);

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

                dst[y, x] =
                    src[y0, x0] * wy0 * wx0 +
                    src[y0, x1] * wy0 * wx1 +
                    src[y1, x0] * wy1 * wx0 +
                    src[y1, x1] * wy1 * wx1;
            }
        }
        return dst;
    }

    private static float[,] GaussianBlur(float[,] src, int kernelSize, float sigma)
    {
        int h = src.GetLength(0), w = src.GetLength(1);
        float[,] dst = new float[h, w];

        // Build kernel
        int radius = kernelSize / 2;
        float[] kernel = new float[kernelSize * kernelSize];
        float sum = 0;
        for (int ky = -radius; ky <= radius; ky++)
        {
            for (int kx = -radius; kx <= radius; kx++)
            {
                float val = MathF.Exp(-(kx * kx + ky * ky) / (2 * sigma * sigma));
                kernel[(ky + radius) * kernelSize + (kx + radius)] = val;
                sum += val;
            }
        }
        for (int i = 0; i < kernel.Length; i++) kernel[i] /= sum;

        // Convolve (reflect border)
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float val = 0;
                for (int ky = -radius; ky <= radius; ky++)
                {
                    for (int kx = -radius; kx <= radius; kx++)
                    {
                        int sy = Math.Clamp(y + ky, 0, h - 1);
                        int sx = Math.Clamp(x + kx, 0, w - 1);
                        val += src[sy, sx] * kernel[(ky + radius) * kernelSize + (kx + radius)];
                    }
                }
                dst[y, x] = val;
            }
        }
        return dst;
    }

    // >.net9 tensor
    private static float[] NormalizePerPatch(float[] patches, int embedDim)
    {
        float[] normed = new float[patches.Length];
        for (int p = 0; p < patches.Length / embedDim; p++)
        {
            var src = patches.AsSpan(p * embedDim, embedDim);
            float norm = MathF.Sqrt(TensorPrimitives.Dot(src, src)) + 1e-8f;
            var dst = normed.AsSpan(p * embedDim, embedDim);
            for (int i = 0; i < embedDim; i++)
                dst[i] = src[i] / norm;
        }
        return normed;
    }

    /// <summary>
    /// 返回 values 中最大的 k 个元素的索引，结果按降序排列。
    /// 使用最小堆，时间复杂度 O(N log k)
    /// </summary>
    private static int[] TopKIndices(float[] values, int k)
    {
        if (k <= 0) return Array.Empty<int>();
        int n = values.Length;
        k = Math.Min(k, n);

        // 最小堆，元素是 (索引, 值)，按值排序，堆顶是最小值
        var heap = new PriorityQueue<int, float>(k);
        for (int i = 0; i < n; i++)
        {
            float v = values[i];
            if (heap.Count < k)
            {
                heap.Enqueue(i, v);
            }
            else if (v > heap.Peek())
            {
                heap.Dequeue();
                heap.Enqueue(i, v);
            }
        }

        int[] result = new int[k];
        for (int i = k - 1; i >= 0; i--)
            result[i] = heap.Dequeue();
        return result;   // 降序
    }

    #endregion

    public void Dispose()
    {
        _session?.Dispose();
    }
}
