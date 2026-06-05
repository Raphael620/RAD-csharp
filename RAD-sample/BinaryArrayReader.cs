using System;
using System.IO;

public static class BinaryArrayReader
{
    /// <summary>读取二维 float32 数组 [rows, cols]</summary>
    public static float[,] Read2DArray(string filePath)
    {
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(fs);

        int ndim = reader.ReadInt32();
        if (ndim != 2) throw new Exception($"期望二维数组，实际维度 {ndim}");

        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        float[,] array = new float[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                array[i, j] = reader.ReadSingle();
        return array;
    }

    /// <summary>读取三维 float32 数组 [d1, d2, d3]</summary>
    public static float[,,] Read3DArray(string filePath)
    {
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(fs);

        int ndim = reader.ReadInt32();
        if (ndim != 3) throw new Exception($"期望三维数组，实际维度 {ndim}");

        int d1 = reader.ReadInt32();
        int d2 = reader.ReadInt32();
        int d3 = reader.ReadInt32();
        float[,,] array = new float[d1, d2, d3];
        for (int i = 0; i < d1; i++)
            for (int j = 0; j < d2; j++)
                for (int k = 0; k < d3; k++)
                    array[i, j, k] = reader.ReadSingle();
        return array;
    }

    /// <summary>读取 layers.bin，返回 int 数组</summary>
    public static int[] ReadLayers(string filePath)
    {
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(fs);
        int count = reader.ReadInt32();
        int[] layers = new int[count];
        for (int i = 0; i < count; i++)
            layers[i] = reader.ReadInt32();
        return layers;
    }
}