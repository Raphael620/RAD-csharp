using System.Runtime.InteropServices;

namespace RAD.Detector;

/// <summary>
/// Reads and writes the RAD memory bank in binary format compatible with bank_pth2bin.py.
/// Format: [int32 ndim][int32 dim0][int32 dim1]...[float32[] data]
/// </summary>
public static class BankSerializer
{
    public static void SaveFloatArray(BinaryWriter writer, float[] data, int[] shape)
    {
        writer.Write(shape.Length);              // ndim
        foreach (var dim in shape)
            writer.Write(dim);
        writer.Write(MemoryMarshal.Cast<float, byte>(data));
    }

    public static float[] LoadFloatArray(BinaryReader reader, out int[] shape)
    {
        int ndim = reader.ReadInt32();
        shape = new int[ndim];
        for (int i = 0; i < ndim; i++)
            shape[i] = reader.ReadInt32();

        int total = 1;
        foreach (var d in shape) total *= d;

        byte[] bytes = reader.ReadBytes(total * sizeof(float));
        return MemoryMarshal.Cast<byte, float>(bytes).ToArray();
    }

    // --- 2D (cls) ---
    public static float[] Load2D(string path, out int rows, out int cols)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);
        var data = LoadFloatArray(br, out int[] shape);
        if (shape.Length != 2)
            throw new InvalidDataException($"Expected 2D array, got {shape.Length}D at {path}");
        rows = shape[0];
        cols = shape[1];
        return data;
    }

    // --- 3D (patch) ---
    public static float[] Load3D(string path, out int d0, out int d1, out int d2)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);
        var data = LoadFloatArray(br, out int[] shape);
        if (shape.Length != 3)
            throw new InvalidDataException($"Expected 3D array, got {shape.Length}D at {path}");
        d0 = shape[0];
        d1 = shape[1];
        d2 = shape[2];
        return data;
    }

    // --- Layer indices ---
    public static int[] LoadLayers(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);
        int count = br.ReadInt32();
        int[] layers = new int[count];
        for (int i = 0; i < count; i++)
            layers[i] = br.ReadInt32();
        return layers;
    }

    // --- Bank save ---
    public static void SaveBank(
        string bankDir, int[] layers,
        float[][] clsBanks,       // [numLayers][N * C]
        float[][] patchBanks,     // [numLayers][N * L * C] flat
        int embedDim, int numPatches)
    {
        Directory.CreateDirectory(bankDir);

        // Save layers.bin
        using (var fs = File.Create(Path.Combine(bankDir, "layers.bin")))
        using (var bw = new BinaryWriter(fs))
        {
            bw.Write(layers.Length);
            foreach (var l in layers) bw.Write(l);
        }

        for (int li = 0; li < layers.Length; li++)
        {
            int layerIdx = layers[li];
            float[] clsData = clsBanks[li];
            float[] patchData = patchBanks[li];

            int N = clsData.Length / embedDim;

            // cls: [N, C]
            {
                using var fs = File.Create(Path.Combine(bankDir, $"cls_layer{layerIdx}.bin"));
                using var bw = new BinaryWriter(fs);
                SaveFloatArray(bw, clsData, [N, embedDim]);
            }

            // patch: [N, L, C]
            {
                using var fs = File.Create(Path.Combine(bankDir, $"patch_layer{layerIdx}.bin"));
                using var bw = new BinaryWriter(fs);
                int C = patchData.Length / (N * numPatches);
                SaveFloatArray(bw, patchData, [N, numPatches, C]);
            }
        }
    }

    /// <summary>
    /// Load a complete bank from directory.
    /// Returns layers, clsBanks as [numLayers][N*C], patchBanks as [numLayers][N*L*C] flat.
    /// </summary>
    public static (int[] layers, float[][] clsBanks, float[][] patchBanks)
        LoadBank(string bankDir)
    {
        var layers = LoadLayers(Path.Combine(bankDir, "layers.bin"));

        var clsBanks = new float[layers.Length][];
        var patchBanks = new float[layers.Length][];

        for (int i = 0; i < layers.Length; i++)
        {
            int layer = layers[i];
            clsBanks[i] = Load2D(Path.Combine(bankDir, $"cls_layer{layer}.bin"), out _, out _);
            patchBanks[i] = Load3D(Path.Combine(bankDir, $"patch_layer{layer}.bin"), out _, out _, out _);
        }

        return (layers, clsBanks, patchBanks);
    }
}
