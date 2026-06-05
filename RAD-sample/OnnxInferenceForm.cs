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
using System.Linq;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;

namespace RAD_WinForms
{
    public partial class OnnxInferenceForm : Form
    {
        private RADInference rad;
        private string selectedImagePath;

        public OnnxInferenceForm()
        {
            InitializeComponent();
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
                btnBatch.Enabled = true;   // 启用批量按钮
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
                lblTime.Text = $"耗时：{sw.ElapsedMilliseconds}ms";
                bool finalResult = score < (float)decimalUpDown1.Value;
                lblOKNG.Text = finalResult ? "正常" : "异常";
                lblOKNG.ForeColor = finalResult ? Color.LimeGreen : Color.Red;
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

        // ========== 批量推理 ==========
        private async void BtnBatch_Click(object sender, EventArgs e)
        {
            if (rad == null)
            {
                MessageBox.Show("模型未加载完成");
                return;
            }

            // 选择图片文件夹
            using (var fbd = new FolderBrowserDialog())
            {
                fbd.Description = "选择包含图片的文件夹";
                if (fbd.ShowDialog() != DialogResult.OK)
                    return;
                string inputDir = fbd.SelectedPath;
                // 选择输出目录
                using (var fbdOut = new FolderBrowserDialog())
                {
                    fbdOut.Description = "选择结果保存文件夹";
                    if (fbdOut.ShowDialog() != DialogResult.OK)
                        return;
                    string outputDir = fbdOut.SelectedPath;

                    // 获取所有图片文件（支持常见格式）
                    var extensions = new[] { "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff" };
                    var files = extensions.SelectMany(ext => Directory.GetFiles(inputDir, ext, SearchOption.TopDirectoryOnly))
                                          .ToList();
                    if (files.Count == 0)
                    {
                        MessageBox.Show("所选文件夹中没有图片文件");
                        return;
                    }

                    // 准备进度显示
                    progressBar.Maximum = files.Count;
                    progressBar.Value = 0;
                    lblProgress.Text = "准备批量推理...";
                    btnBatch.Enabled = false;
                    btnDetect.Enabled = false;

                    try
                    {
                        await Task.Run(() => BatchInferenceAsync(files, outputDir));
                        MessageBox.Show($"批量推理完成！结果保存在：{outputDir}");
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show($"批量推理出错：{ex.Message}");
                    }
                    finally
                    {
                        btnBatch.Enabled = true;
                        btnDetect.Enabled = true;
                        progressBar.Value = 0;
                        lblProgress.Text = "";
                    }
                }
            }
        }

        private async Task BatchInferenceAsync(List<string> imageFiles, string outputDir)
        {
            for (int i = 0; i < imageFiles.Count; i++)
            {
                string file = imageFiles[i];
                string fileName = Path.GetFileNameWithoutExtension(file);
                string outputPath = Path.Combine(outputDir, $"{fileName}_result.png");

                // 更新UI进度（必须在UI线程）
                this.Invoke((Action)(() =>
                {
                    progressBar.Value = i + 1;
                    lblProgress.Text = $"处理中：{Path.GetFileName(file)} ({i + 1}/{imageFiles.Count})";
                }));

                try
                {
                    Stopwatch sw = Stopwatch.StartNew();
                    // 推理
                    var (anomalyMap, score) = rad.Infer(file, outputSize: 256, applySmoothing: true);
                    sw.Stop();
                    long elapsedMs = sw.ElapsedMilliseconds;

                    // 生成可视化图
                    using (var img = Cv2.ImRead(file))
                    {
                        var (heatmapBmp, _, rawBmp) = GenerateVisualization(img, anomalyMap);
                        // 原图 Bitmap
                        using (var originalBmp = new Bitmap(file))
                        {
                            string resultText = score < (float)decimalUpDown1.Value ? "Normal" : "Abnormal";
                            Bitmap combined = CombineImages(originalBmp, heatmapBmp, rawBmp, score, resultText, elapsedMs);
                            combined.Save(outputPath, ImageFormat.Png);
                            combined.Dispose();
                        }
                        heatmapBmp.Dispose();
                        rawBmp.Dispose();
                    }
                }
                catch (Exception ex)
                {
                    // 记录错误但继续处理下一张
                    Debug.WriteLine($"处理 {file} 失败：{ex.Message}");
                    // 可选：在UI上显示错误
                    this.Invoke((Action)(() => lblProgress.Text = $"处理 {Path.GetFileName(file)} 失败"));
                }
            }
        }

        /// <summary>
        /// 将原图、异常热图、Mask图水平拼接，并在右上角绘制文本信息
        /// </summary>
        private Bitmap CombineImages(Bitmap original, Bitmap anomalyMap, Bitmap mask, float score, string resultText, long timeMs)
        {
            // 转为 Mat
            using (Mat originalMat = BitmapConverter.ToMat(original))
            using (Mat anomalyMat = BitmapConverter.ToMat(anomalyMap))
            using (Mat maskMat = BitmapConverter.ToMat(mask))
            {
                // 统一高度（以原图高度为准）
                int targetHeight = originalMat.Height;
                int targetWidthOriginal = originalMat.Width;

                // 缩放异常图和 mask 图到相同高度（保持宽高比）
                double scaleAnomaly = (double)targetHeight / anomalyMat.Height;
                int newWidthAnomaly = (int)(anomalyMat.Width * scaleAnomaly);
                Mat resizedAnomaly = new Mat();
                Cv2.Resize(anomalyMat, resizedAnomaly, new OpenCvSharp.Size(newWidthAnomaly, targetHeight));

                double scaleMask = (double)targetHeight / maskMat.Height;
                int newWidthMask = (int)(maskMat.Width * scaleMask);
                Mat resizedMask = new Mat();
                Cv2.Resize(maskMat, resizedMask, new OpenCvSharp.Size(newWidthMask, targetHeight));

                // 水平拼接
                int totalWidth = targetWidthOriginal + newWidthAnomaly + newWidthMask;
                Mat combined = new Mat(targetHeight, totalWidth, originalMat.Type());
                // 将三张图拷贝到相应区域
                originalMat.CopyTo(combined[new Rect(0, 0, targetWidthOriginal, targetHeight)]);
                resizedAnomaly.CopyTo(combined[new Rect(targetWidthOriginal, 0, newWidthAnomaly, targetHeight)]);
                resizedMask.CopyTo(combined[new Rect(targetWidthOriginal + newWidthAnomaly, 0, newWidthMask, targetHeight)]);

                // 绘制文本（右上角）
                string text = $"Score: {score:F4}  Result: {resultText}  Time: {timeMs}ms";
                // 计算文本大小（OpenCvSharp 中 GetTextSize 需要字体和厚度）
                double fontScale = 0.8;
                int thickness = 2;
                var fontFace = HersheyFonts.HersheySimplex;
                OpenCvSharp.Size textSize = Cv2.GetTextSize(text, fontFace, fontScale, thickness, out int baseline);
                // 文本位置：右上角留出 10 像素边距
                OpenCvSharp.Point textOrg = new OpenCvSharp.Point(combined.Width - textSize.Width - 10, textSize.Height + 10);
                // 绘制黑色阴影
                Cv2.PutText(combined, text, new OpenCvSharp.Point(textOrg.X + 1, textOrg.Y + 1), fontFace, fontScale, Scalar.Black, thickness);
                // 绘制白色文字
                Cv2.PutText(combined, text, textOrg, fontFace, fontScale, Scalar.White, thickness);

                // 转回 Bitmap
                Bitmap result = BitmapConverter.ToBitmap(combined);

                // 释放临时 Mat
                resizedAnomaly.Dispose();
                resizedMask.Dispose();
                combined.Dispose();

                return result;
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
            if (range < (float)decimalUpDown1.Value) range = 1;

            // 归一化并取反用于热图
            byte[,] amByte = new byte[h, w];
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    byte gray = (byte)((anomalyMap[i, j] - min) / range * 255);
                    amByte[i, j] = (byte)(255 - gray);
                }
            }

            using Mat heatmapMat = new Mat(h, w, MatType.CV_8UC1);
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                    heatmapMat.Set(i, j, amByte[i, j]);

            using Mat colorMap = new Mat();
            Cv2.ApplyColorMap(heatmapMat, colorMap, ColormapTypes.Jet);
            Cv2.CvtColor(colorMap, colorMap, ColorConversionCodes.BGR2RGB);

            using Mat colorMapResized = new Mat();
            Cv2.Resize(colorMap, colorMapResized, new OpenCvSharp.Size(img.Width, img.Height));

            using Mat overlayMat = new Mat();
            Cv2.AddWeighted(img, 0.5, colorMapResized, 0.5, 0, overlayMat);

            // 二值 mask
            using Mat maskMat = new Mat(h, w, MatType.CV_8UC1);
            float threshold = 0.5f;
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    float normValue = (anomalyMap[i, j] - min) / range;
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