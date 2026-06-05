using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace RAD.Detector;

public static class ImagePreprocessor
{
    private const int ResizeSize = 512;
    private const int CropSize = 448;

    private const float MeanR = 0.485f, MeanG = 0.456f, MeanB = 0.406f;
    private const float StdR = 0.229f, StdG = 0.224f, StdB = 0.225f;

    /// <summary>
    /// Load and decode an image, converting to Rgb24 regardless of source format.
    /// Handles JPG/PNG/BMP, 8/24/32bit, CMYK, grayscale, etc.
    /// </summary>
    private static Image<Rgb24> LoadRgb24(string filePath)
    {
        //using var raw = Image.Load(filePath);
        //return raw.CloneAs<Rgb24>();
        return Image.Load<Rgb24>(filePath);
    }

    public static float[] Preprocess(string imagePath)
    {
        using var image = LoadRgb24(imagePath);
        //image.Mutate(ctx => ctx.Resize(ResizeSize, ResizeSize));
        int offset = (ResizeSize - CropSize) / 2;
        //image.Mutate(ctx => ctx.Crop(new Rectangle(offset, offset, CropSize, CropSize)));

        image.Mutate(ctx => ctx
            .Resize(ResizeSize, ResizeSize)
            .Crop(new Rectangle(offset, offset, CropSize, CropSize)));
        return ToNchwTensor(image);
    }

    public static Image<Rgb24> LoadForDisplay(string imagePath)
    {
        return LoadRgb24(imagePath);
    }

    public static float[] ToNchwTensor0(Image<Rgb24> image)
    {
        int w = image.Width, h = image.Height;
        int ch = w * h;
        float[] t = new float[3 * ch];

        image.ProcessPixelRows(acc =>
        {
            for (int y = 0; y < h; y++)
            {
                Span<Rgb24> row = acc.GetRowSpan(y);
                int off = y * w;
                for (int x = 0; x < w; x++)
                {
                    int i = off + x;
                    t[i] = (row[x].R / 255f - MeanR) / StdR;
                    t[ch + i] = (row[x].G / 255f - MeanG) / StdG;
                    t[2 * ch + i] = (row[x].B / 255f - MeanB) / StdB;
                }
            }
        });

        return t;
    }
    public static float[] ToNchwTensor(Image<Rgb24> image)
    {
        int w = image.Width, h = image.Height;
        int ch = w * h;
        float[] t = new float[3 * ch];

        // 预计算： pixel * scale + bias   (原式: (pixel/255 - mean)/std)
        // scale = 1/(255*std),  bias = -mean/std
        float scaleR = 1.0f / (255f * StdR), biasR = -MeanR / StdR;
        float scaleG = 1.0f / (255f * StdG), biasG = -MeanG / StdG;
        float scaleB = 1.0f / (255f * StdB), biasB = -MeanB / StdB;

        image.ProcessPixelRows(acc =>
        {
            for (int y = 0; y < h; y++)
            {
                Span<Rgb24> row = acc.GetRowSpan(y);
                int off = y * w;
                for (int x = 0; x < w; x++)
                {
                    int i = off + x;
                    var px = row[x];
                    t[i] = px.R * scaleR + biasR;
                    t[ch + i] = px.G * scaleG + biasG;
                    t[2 * ch + i] = px.B * scaleB + biasB;
                }
            }
        });

        return t;
    }
}
