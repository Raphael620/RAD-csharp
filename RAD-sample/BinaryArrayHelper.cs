using System;
using System.IO;
using System.Runtime.InteropServices;

public static class BinaryArrayHelper
{
    public static void Save2DArray(float[,] array, string filePath)
    {
        int rows = array.GetLength(0);
        int cols = array.GetLength(1);
        using var fs = new FileStream(filePath, FileMode.Create);
        using var writer = new BinaryWriter(fs);
        writer.Write(2); // ndim
        writer.Write(rows);
        writer.Write(cols);
        int total = rows * cols;
        var data = new float[total];
        Buffer.BlockCopy(array, 0, data, 0, total * sizeof(float));
        writer.Write(MemoryMarshal.AsBytes(data.AsSpan()));
    }

    public static void Save3DArray(float[,,] array, string filePath)
    {
        int d1 = array.GetLength(0);
        int d2 = array.GetLength(1);
        int d3 = array.GetLength(2);
        using var fs = new FileStream(filePath, FileMode.Create);
        using var writer = new BinaryWriter(fs);
        writer.Write(3);
        writer.Write(d1);
        writer.Write(d2);
        writer.Write(d3);
        int total = d1 * d2 * d3;
        var data = new float[total];
        Buffer.BlockCopy(array, 0, data, 0, total * sizeof(float));
        writer.Write(MemoryMarshal.AsBytes(data.AsSpan()));
    }

    public static void SaveLayers(int[] layers, string filePath)
    {
        using var fs = new FileStream(filePath, FileMode.Create);
        using var writer = new BinaryWriter(fs);
        writer.Write(layers.Length);
        foreach (var l in layers)
            writer.Write(l);
    }

    public static float[,] Read2DArray(string filePath)
    {
        using var fs = new FileStream(filePath, FileMode.Open);
        using var reader = new BinaryReader(fs);
        int ndim = reader.ReadInt32();
        if (ndim != 2) throw new Exception("Not a 2D array");
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        int total = rows * cols;
        var data = new float[total];
        var bytes = reader.ReadBytes(total * sizeof(float));
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        var result = new float[rows, cols];
        Buffer.BlockCopy(data, 0, result, 0, data.Length * sizeof(float));
        return result;
    }

    public static float[,,] Read3DArray(string filePath)
    {
        using var fs = new FileStream(filePath, FileMode.Open);
        using var reader = new BinaryReader(fs);
        int ndim = reader.ReadInt32();
        if (ndim != 3) throw new Exception("Not a 3D array");
        int d1 = reader.ReadInt32();
        int d2 = reader.ReadInt32();
        int d3 = reader.ReadInt32();
        int total = d1 * d2 * d3;
        var data = new float[total];
        var bytes = reader.ReadBytes(total * sizeof(float));
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        var result = new float[d1, d2, d3];
        Buffer.BlockCopy(data, 0, result, 0, data.Length * sizeof(float));
        return result;
    }

    public static int[] ReadLayers(string filePath)
    {
        using var fs = new FileStream(filePath, FileMode.Open);
        using var reader = new BinaryReader(fs);
        int count = reader.ReadInt32();
        var layers = new int[count];
        for (int i = 0; i < count; i++)
            layers[i] = reader.ReadInt32();
        return layers;
    }
}