using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace RAD.Detector;

public static class VisualizationHelper
{
    private static readonly Rgba32[] JetLut = BuildJetLut();

    private static Rgba32[] BuildJetLut()
    {
        var lut = new Rgba32[256];
        for (int i = 0; i < 256; i++)
        {
            float t = i / 255f;
            byte r, g, b;
            if (t < 0.125f)      { r = 0; g = 0; b = (byte)(128 + 127 * t / 0.125f); }
            else if (t < 0.375f) { r = 0; g = (byte)(255 * (t - 0.125f) / 0.25f); b = 255; }
            else if (t < 0.625f) { r = (byte)(255 * (t - 0.375f) / 0.25f); g = 255; b = (byte)(255 * (1 - (t - 0.375f) / 0.25f)); }
            else if (t < 0.875f) { r = 255; g = (byte)(255 * (1 - (t - 0.625f) / 0.25f)); b = 0; }
            else                 { r = (byte)(255 * (1 - (t - 0.875f) / 0.25f * 0.5f)); g = 0; b = 0; }
            lut[i] = new Rgba32(r, g, b, 255);
        }
        return lut;
    }

    [Obsolete("保留用于性能对比")]
    public static Image<Rgba32> CreateHeatmap0(float[,] amap, int targetW, int targetH)
    {
        int h = amap.GetLength(0), w = amap.GetLength(1);

        // Use fixed scale: 0 -> blue (cold, normal), max(0.5, anomalyScore) -> red (hot, anomaly)
        // This gives consistent coloring across all images instead of per-image min/max
        float vmax = 0f;
        foreach (var v in amap) if (v > vmax) vmax = v;
        // Floor at 0.5 so normal images don't look noisy from microscopic variations
        float scaleMax = Math.Max(vmax, 0.3f);
        float invScale = 255f / scaleMax;

        var img = new Image<Rgba32>(targetW, targetH);
        float stepX = (float)(w - 1) / Math.Max(targetW - 1, 1);
        float stepY = (float)(h - 1) / Math.Max(targetH - 1, 1);

        img.ProcessPixelRows(acc =>
        {
            for (int y = 0; y < targetH; y++)
            {
                Span<Rgba32> row = acc.GetRowSpan(y);
                int sy = (int)(y * stepY + 0.5f);
                for (int x = 0; x < targetW; x++)
                {
                    int sx = (int)(x * stepX + 0.5f);
                    float v = amap[sy, sx] * invScale;
                    int idx = v < 0 ? 0 : v > 255 ? 255 : (int)v;
                    row[x] = JetLut[idx];
                }
            }
        });
        return img;
    }

    [Obsolete("保留用于性能对比")]
    public static Image<Rgba32> CreateOverlay0(Image<Rgb24> orig, Image<Rgba32> heat, float alpha = 0.5f)
    {
        int w = orig.Width, h = orig.Height;
        if (heat.Width != w || heat.Height != h)
            heat.Mutate(ctx => ctx.Resize(w, h));

        var origPix = new Rgb24[w * h];
        orig.CopyPixelDataTo(origPix);
        var heatPix = new Rgba32[w * h];
        heat.CopyPixelDataTo(heatPix);

        var outImg = new Image<Rgba32>(w, h);
        float ia = 1f - alpha;

        outImg.ProcessPixelRows(acc =>
        {
            for (int y = 0; y < h; y++)
            {
                Span<Rgba32> dst = acc.GetRowSpan(y);
                int off = y * w;
                for (int x = 0; x < w; x++)
                {
                    int i = off + x;
                    var o = origPix[i]; var hm = heatPix[i];
                    dst[x] = new Rgba32(
                        (byte)(o.R * ia + hm.R * alpha),
                        (byte)(o.G * ia + hm.G * alpha),
                        (byte)(o.B * ia + hm.B * alpha),
                        255);
                }
            }
        });
        return outImg;
    }

    public static Image<Rgba32> CreateBinaryMask(float[,] amap, float thresh, int targetW, int targetH)
    {
        int ah = amap.GetLength(0), aw = amap.GetLength(1);
        var mask = new Image<Rgba32>(targetW, targetH);
        float stepX = (float)(aw - 1) / Math.Max(targetW - 1, 1);
        float stepY = (float)(ah - 1) / Math.Max(targetH - 1, 1);

        mask.ProcessPixelRows(acc =>
        {
            for (int y = 0; y < targetH; y++)
            {
                Span<Rgba32> row = acc.GetRowSpan(y);
                int ay = (int)(y * stepY + 0.5f);
                for (int x = 0; x < targetW; x++)
                {
                    int ax = (int)(x * stepX + 0.5f);
                    row[x] = amap[ay, ax] > thresh
                        ? new Rgba32(255, 255, 255, 255)
                        : new Rgba32(0, 0, 0, 255);
                }
            }
        });
        return mask;
    }

    public static byte[] ToPngBytes(Image<Rgba32> image)
    {
        using var ms = new MemoryStream();
        image.SaveAsPng(ms);
        return ms.ToArray();
    }

    [Obsolete("保留用于性能对比")]
    public static Image<Rgba32> ResizeForDisplay0(Image<Rgba32> src, int maxSide)
    {
        float scale = Math.Min((float)maxSide / src.Width, (float)maxSide / src.Height);
        int nw = (int)(src.Width * scale), nh = (int)(src.Height * scale);

        var srcPix = new Rgba32[src.Width * src.Height];
        src.CopyPixelDataTo(srcPix);

        var result = new Image<Rgba32>(nw, nh);
        float stepX = (float)(src.Width - 1) / Math.Max(nw - 1, 1);
        float stepY = (float)(src.Height - 1) / Math.Max(nh - 1, 1);

        result.ProcessPixelRows(acc =>
        {
            for (int y = 0; y < nh; y++)
            {
                Span<Rgba32> dst = acc.GetRowSpan(y);
                int sy = (int)(y * stepY + 0.5f);
                int srcOff = sy * src.Width;
                for (int x = 0; x < nw; x++)
                {
                    int sx = (int)(x * stepX + 0.5f);
                    dst[x] = srcPix[srcOff + sx];
                }
            }
        });
        return result;
    }




    public static Image<Rgba32> CreateOverlay(Image<Rgb24> orig, Image<Rgba32> heat, float alpha = 0.5f)
    {
        int w = orig.Width, h = orig.Height;
        if (heat.Width != w || heat.Height != h)
            heat.Mutate(ctx => ctx.Resize(w, h));

        // 一次性复制到数组，对于 448×448 来说内存开销很小
        var origPix = new Rgb24[w * h];
        orig.CopyPixelDataTo(origPix);
        var heatPix = new Rgba32[w * h];
        heat.CopyPixelDataTo(heatPix);

        var outImg = new Image<Rgba32>(w, h);
        float ia = 1f - alpha;

        outImg.ProcessPixelRows(acc =>
        {
            for (int y = 0; y < h; y++)
            {
                Span<Rgba32> dst = acc.GetRowSpan(y);
                int off = y * w;
                for (int x = 0; x < w; x++)
                {
                    int i = off + x;
                    var o = origPix[i];
                    var hm = heatPix[i];
                    dst[x] = new Rgba32(
                        (byte)(o.R * ia + hm.R * alpha),
                        (byte)(o.G * ia + hm.G * alpha),
                        (byte)(o.B * ia + hm.B * alpha),
                        255);
                }
            }
        });
        return outImg;
    }
    public static Image<Rgba32> ResizeForDisplay(Image<Rgba32> src, int maxSide)
    {
        float scale = Math.Min((float)maxSide / src.Width, (float)maxSide / src.Height);
        int nw = (int)(src.Width * scale), nh = (int)(src.Height * scale);

        // 复制原图像素到数组（避免 ref struct 嵌套问题）
        var srcPix = new Rgba32[src.Width * src.Height];
        src.CopyPixelDataTo(srcPix);

        var result = new Image<Rgba32>(nw, nh);
        float stepX = (float)(src.Width - 1) / Math.Max(nw - 1, 1);
        float stepY = (float)(src.Height - 1) / Math.Max(nh - 1, 1);

        result.ProcessPixelRows(acc =>
        {
            for (int y = 0; y < nh; y++)
            {
                int sy = (int)(y * stepY + 0.5f);
                Span<Rgba32> dst = acc.GetRowSpan(y);
                int srcOff = sy * src.Width;
                for (int x = 0; x < nw; x++)
                {
                    int sx = (int)(x * stepX + 0.5f);
                    dst[x] = srcPix[srcOff + sx];
                }
            }
        });
        return result;
    }

    public static Image<Rgba32> CreateHeatmap(float[,] amap, int targetW, int targetH)
    {
        int h = amap.GetLength(0), w = amap.GetLength(1);

        float vmax = 0f;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                if (amap[y, x] > vmax) vmax = amap[y, x];
        float scaleMax = Math.Max(vmax, 0.3f);
        float invScale = 255f / scaleMax;

        var img = new Image<Rgba32>(targetW, targetH);
        float stepX = (float)(w - 1) / Math.Max(targetW - 1, 1);
        float stepY = (float)(h - 1) / Math.Max(targetH - 1, 1);

        img.ProcessPixelRows(acc =>
        {
            for (int y = 0; y < targetH; y++)
            {
                Span<Rgba32> row = acc.GetRowSpan(y);
                int sy = (int)(y * stepY + 0.5f);
                for (int x = 0; x < targetW; x++)
                {
                    int sx = (int)(x * stepX + 0.5f);
                    float v = amap[sy, sx] * invScale;
                    int idx = v < 0 ? 0 : v > 255 ? 255 : (int)v;
                    row[x] = JetLut[idx];
                }
            }
        });
        return img;
    }
}
