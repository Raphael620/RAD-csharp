using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace RADTool
{
    public class common
    {
        //Convert the image to NCHW
        public static float[] ExtractMat(Mat src)
        {
            OpenCvSharp.Size size = src.Size();
            int channels = src.Channels();
            float[] result = new float[size.Width * size.Height * channels];
            GCHandle resultHandle = default;
            try
            {
                resultHandle = GCHandle.Alloc(result, GCHandleType.Pinned);
                IntPtr resultPtr = resultHandle.AddrOfPinnedObject();
                for (int i = 0; i < channels; ++i)
                {
                    Mat cmat = Mat.FromPixelData(
                      src.Height, src.Width,
                      MatType.CV_32FC1,
                      resultPtr + i * size.Width * size.Height * sizeof(float));

                    Cv2.ExtractChannel(src, cmat, i);

                    cmat.Dispose();

                }
            }
            finally
            {
                resultHandle.Free();
            }

            return result;
        }
    }
}
