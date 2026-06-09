using Microsoft.ML.OnnxRuntime;

namespace RAD.Detector;

public sealed class OnnxModelInfo
{
    public int[] LayerIndices { get; }
    public int EmbedDim { get; }
    public int NumPatches { get; }
    public int PatchH { get; }
    public int PatchW { get; }
    /// <summary>The ONNX input tensor name (e.g. "input" for DINOv3, "pixel_values" for DINOv2).</summary>
    public string InputName { get; }
    /// <summary>True for DINOv3-style multi-layer models (patch_X/cls_X outputs), false for single-output models.</summary>
    public bool IsMultiLayer { get; }

    private OnnxModelInfo(int[] layers, int embedDim, int numPatches, string inputName, bool isMultiLayer)
    {
        LayerIndices = layers;
        EmbedDim = embedDim;
        NumPatches = numPatches;
        InputName = inputName;
        IsMultiLayer = isMultiLayer;
        int sqrtPatches = (int)Math.Sqrt(numPatches);
        PatchH = sqrtPatches;
        PatchW = sqrtPatches;
    }

    /// <summary>
    /// Extract model info by running a dummy inference to get actual output shapes.
    /// Supports both multi-layer models (DINOv3: patch_X/cls_X outputs) and single-output
    /// models (DINOv2: last_hidden_state).
    /// </summary>
    public static OnnxModelInfo FromSession(InferenceSession session)
    {
        // --- Detect input name ---
        string inputName = "input"; // fallback
        if (session.InputMetadata.Count > 0)
            inputName = session.InputMetadata.First().Key;

        // --- Discover layer indices from output names ---
        var layerSet = new HashSet<int>();
        foreach (var name in session.OutputMetadata.Keys)
        {
            if (name.StartsWith("patch_") || name.StartsWith("cls_"))
            {
                var idxStr = name.Split('_')[1];
                if (int.TryParse(idxStr, out int idx))
                    layerSet.Add(idx);
            }
        }

        bool isMultiLayer = layerSet.Count > 0;

        // Run a dummy inference to get actual output dimensions
        int[] dims = [1, 3, 448, 448];
        var dummyTensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(dims);
        var namedInput = NamedOnnxValue.CreateFromTensor(inputName, dummyTensor);
        var inputs = new List<NamedOnnxValue> { namedInput };

        using var results = session.Run(inputs);

        int embedDim = 0, numPatches = 0;
        int[] layers;

        if (isMultiLayer)
        {
            // DINOv3-style: outputs are named patch_3, cls_3, patch_6, cls_6, ...
            layers = layerSet.OrderBy(x => x).ToArray();
            foreach (var result in results)
            {
                if (result.Name == $"patch_{layers[0]}")
                {
                    var t = result.AsTensor<float>();
                    numPatches = (int)t.Dimensions[1];
                    embedDim = (int)t.Dimensions[2];
                    break;
                }
            }
        }
        else
        {
            // DINOv2-style: single output last_hidden_state [B, 1 + numPatches, embedDim]
            layers = [0];
            foreach (var result in results)
            {
                var t = result.AsTensor<float>();
                int totalTokens = (int)t.Dimensions[1];   // 1 (CLS) + L (patches)
                numPatches = totalTokens - 1;
                embedDim = (int)t.Dimensions[2];
                break;
            }
        }

        return new OnnxModelInfo(layers, embedDim, numPatches, inputName, isMultiLayer);
    }
}
