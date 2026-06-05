using Microsoft.ML.OnnxRuntime;

namespace RAD.Detector;

public sealed class OnnxModelInfo
{
    public int[] LayerIndices { get; }
    public int EmbedDim { get; }
    public int NumPatches { get; }
    public int PatchH { get; }
    public int PatchW { get; }

    private OnnxModelInfo(int[] layers, int embedDim, int numPatches)
    {
        LayerIndices = layers;
        EmbedDim = embedDim;
        NumPatches = numPatches;
        int sqrtPatches = (int)Math.Sqrt(numPatches);
        PatchH = sqrtPatches;
        PatchW = sqrtPatches;
    }

    /// <summary>
    /// Extract model info by running a dummy inference to get actual output shapes.
    /// Output metadata may have dynamic dims; we run with real input to get concrete sizes.
    /// </summary>
    public static OnnxModelInfo FromSession(InferenceSession session)
    {
        // Discover layer indices from output names
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

        var layers = layerSet.OrderBy(x => x).ToArray();

        // Run a dummy inference to get actual output dimensions
        int[] dims = [1, 3, 448, 448];
        var dummyTensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(dims);
        var namedInput = NamedOnnxValue.CreateFromTensor("input", dummyTensor);
        var inputs = new List<NamedOnnxValue> { namedInput };

        using var results = session.Run(inputs);

        int embedDim = 0, numPatches = 0;
        foreach (var result in results)
        {
            if (result.Name == $"patch_{layers[0]}")
            {
                var t = result.AsTensor<float>();
                var actualDims = t.Dimensions; // [1, L, C]
                numPatches = (int)actualDims[1];
                embedDim = (int)actualDims[2];
                break;
            }
        }

        return new OnnxModelInfo(layers, embedDim, numPatches);
    }
}
