using OpenCvSharp;
using OpenCvSharp.Extensions;
using System;
using System.Drawing;
using System.Windows.Forms;
using System.IO;
using System.Threading.Tasks;
using MathNet.Numerics;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace RAD_WinForms
{
    public partial class TestForm : Form
    {
        private RADInference rad;
        private string selectedImagePath;

        public TestForm()
        {
            InitializeComponent();
            // 异步加载模型，不阻塞 UI 线程
            LoadModelAsync();
            
        }

        private async void LoadModelAsync()
        {
            try
            {
                lblResult.Text = "正在加载模型...";
                string onnxPath = "./Model/dinov3_multilayer.onnx";
                string bankDir = "./Bank";
                rad = new RADInference(onnxPath, bankDir, kImage: 1);
                lblResult.Text = "模型加载完成，请选择图像";
                btnDetect.Enabled = true;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"模型加载失败: {ex.Message}");
                lblResult.Text = "加载失败";
            }
        }

        private void BtnLoadImage_Click(object sender, EventArgs e)
        {
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                selectedImagePath = openFileDialog.FileName;
                pictureBoxOriginal.Image = new Bitmap(selectedImagePath);
                pictureBoxHeatmap.Image = null;
                pictureBoxOverlay.Image = null;
                pictureBoxRaw.Image = null;
                lblResult.Text = "图像已加载，点击检测";
            }
        }

        private async void BtnDetect_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrEmpty(selectedImagePath))
            {
                MessageBox.Show("请先选择图像");
                return;
            }

            btnDetect.Enabled = false;
            lblResult.Text = "检测中...";

            
            var progress = new Progress<string>(msg =>
            {
                lblResult.Text = msg;
            });

            try
            {
                Stopwatch sw = Stopwatch.StartNew();
                var (anomalyMap, score) = await Task.Run(() =>
                    rad.Infer(selectedImagePath, outputSize: 256, applySmoothing: true,
                              progressCallback: msg => ((IProgress<string>)progress).Report(msg)));

                using var img = Cv2.ImRead(selectedImagePath);
                var (heatmapBmp, overlayBmp, rawBmp) = GenerateVisualization(img, anomalyMap);

                pictureBoxHeatmap.Image = heatmapBmp;
                pictureBoxOverlay.Image = overlayBmp;
                pictureBoxRaw.Image = rawBmp;
                lblResult.Text = $"检测完成，异常分数: {score:F4}";
                sw.Stop();
                lblTime.Text=$"耗时：{sw.ElapsedMilliseconds.ToString()}ms";
                bool finalResult = score < (float)decimalUpDown1.Value;
                lblOKNG.Text = finalResult ? "正常" : "异常";
                lblOKNG.ForeColor = finalResult?Color.LimeGreen:Color.Red;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"检测失败: {ex.Message}");
                lblResult.Text = "检测失败";
            }
            finally
            {
                btnDetect.Enabled = true;
                btnDetect.Focus();
            }
        }

        private (Bitmap heatmap, Bitmap overlay, Bitmap raw) GenerateVisualization(Mat img, float[,] anomalyMap)
        {
            int h = anomalyMap.GetLength(0);
            int w = anomalyMap.GetLength(1);

            float min = float.MaxValue, max = float.MinValue;
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                {
                    if (anomalyMap[i, j] < min) min = anomalyMap[i, j];
                    if (anomalyMap[i, j] > max) max = anomalyMap[i, j];
                }
            float range = max - min;
            if (range < (float)decimalUpDown1.Value) range = 1;//1e-6

            // 归一化并取反用于热图
            byte[,] amByte = new byte[h, w];
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    byte gray = (byte)((anomalyMap[i, j] - min) / range * 255);
                    amByte[i, j] = (byte)(255 - gray);  // 取反
                }
            }

            // 生成热图
            using Mat heatmapMat = new Mat(h, w, MatType.CV_8UC1);
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                    heatmapMat.Set(i, j, amByte[i, j]);

            using Mat colorMap = new Mat();
            Cv2.ApplyColorMap(heatmapMat, colorMap, ColormapTypes.Jet);
            Cv2.CvtColor(colorMap, colorMap, ColorConversionCodes.BGR2RGB);

            using Mat colorMapResized = new Mat();
            Cv2.Resize(colorMap, colorMapResized, new OpenCvSharp.Size(img.Width, img.Height));

            // 叠加图
            using Mat overlayMat = new Mat();
            Cv2.AddWeighted(img, 0.5, colorMapResized, 0.5, 0, overlayMat);

            // 生成二值 mask 图
            using Mat maskMat = new Mat(h, w, MatType.CV_8UC1);
            float threshold = 0.5f;
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    float normValue = (anomalyMap[i, j] - min) / range; // 0~1
                    byte val = (normValue > threshold) ? (byte)255 : (byte)0;
                    maskMat.Set(i, j, val);
                }
            }

            using Mat maskColor = new Mat();
            Cv2.CvtColor(maskMat, maskColor, ColorConversionCodes.GRAY2RGB);
            using Mat maskResized = new Mat();
            Cv2.Resize(maskColor, maskResized, new OpenCvSharp.Size(img.Width, img.Height));

            Bitmap heatmapBmp = BitmapConverter.ToBitmap(colorMapResized);
            Bitmap overlayBmp = BitmapConverter.ToBitmap(overlayMat);
            Bitmap rawBmp = BitmapConverter.ToBitmap(maskResized);

            return (heatmapBmp, overlayBmp, rawBmp);
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            rad?.Dispose();
            base.OnFormClosing(e);
        }
    }
}