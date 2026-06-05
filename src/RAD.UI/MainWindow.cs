using Aprillz.MewUI;
using Aprillz.MewUI.Controls;
using Aprillz.MewUI.Resources;
using RAD.Detector;
using SixLabors.ImageSharp.PixelFormats;

namespace RAD.UI;

public class MainWindow
{
    private readonly AppViewModel _vm = new();
    private RadInference? _inference;
    private Window? _window;
    private bool _bankLoaded;

    // Image controls for updating
    private Image? _originalView;
    private Image? _heatmapView;
    private Image? _overlayView;
    private Image? _maskView;

    // Timing
    private double _preMs, _boneMs, _postMs;

    private const int DisplaySize = 256;

    public Window Build()
    {
        // Find icon.ico
        string[] iconCandidates = [
            Path.Combine(Directory.GetCurrentDirectory(), "src", "RAD.UI", "icon.ico"),
            Path.Combine(Directory.GetCurrentDirectory(), "icon.ico"),
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "icon.ico"),
        ];
        string? iconPath = iconCandidates.FirstOrDefault(File.Exists);

        _window = new Window()
            .Resizable(1100, 600)
            .StartCenterScreen()
            .Title("RAD - Anomaly Detection");
        if (iconPath != null)
            _window.Icon = Aprillz.MewUI.IconSource.FromFile(iconPath);
        _window.OnLoaded(() => TryInitModel())
            .OnClosed(() => _inference?.Dispose())
            .Content(BuildContent());

        return _window;
    }

    private Element BuildContent()
    {
        return new DockPanel()
            .LastChildFill()
            .Padding(12)
            .Children(
                BuildTopPanel().DockTop(),
                BuildResultPanel()
            );
    }

    private Element BuildTopPanel()
    {
        return new StackPanel()
            .Vertical()
            .Spacing(8)
            .Children(
                // Row 1: Model path
                new StackPanel()
                    .Horizontal()
                    .Spacing(8)
                    .Children(
                        new Label().Text("Model:").Width(50).CenterVertical(),
                        new TextBox()
                            .Text(_vm.ModelPath.Value)
                            .OnTextChanged(txt =>
                            {
                                if (txt != null)
                                    _vm.ModelPath.Value = txt;
                            })
                            .StretchHorizontal(),
                        new Button()
                            .Content("Browse...")
                            .OnClick(() => BrowseModel())
                    ),

                // Row 2: Device + Threshold
                new StackPanel()
                    .Horizontal()
                    .Spacing(16)
                    .Children(
                        new StackPanel()
                            .Horizontal()
                            .Spacing(4)
                            .Children(
                                new Label().Text("Device:").CenterVertical(),
                                new ComboBox()
                                    .Items(_vm.DeviceOptions)
                                    .BindSelectedIndex(_vm.DeviceIndex)
                                    .OnSelectionChanged(obj =>
                                    {
                                        if (_inference != null)
                                        {
                                            string result = _inference.SwitchDevice(_vm.DeviceOptions[_vm.DeviceIndex.Value]);
                                            _vm.AppendLog($"Switched to {result}");
                                        }
                                    })
                            ),
                        new StackPanel()
                            .Horizontal()
                            .Spacing(4)
                            .Children(
                                new Label().Text("kImage:").CenterVertical(),
                                new TextBox()
                                    .Text(_vm.KImage.Value.ToString())
                                    .Width(50)
                                    .OnTextChanged(txt =>
                                    {
                                        if (int.TryParse(txt, out int val) && val > 0)
                                            _vm.KImage.Value = val;
                                    })
                            ),
                        new StackPanel()
                            .Horizontal()
                            .Spacing(4)
                            .Children(
                                new Label().Text("Threshold:").CenterVertical(),
                                new TextBox()
                                    .Text("0.12")
                                    .Width(60)
                                    .OnTextChanged(txt =>
                                    {
                                        if (float.TryParse(txt, out float val))
                                            _vm.Threshold.Value = val;
                                    })
                            )
                    ),

                // Row 3: Buttons
                new StackPanel()
                    .Horizontal()
                    .Spacing(8)
                    .Children(
                        new Button()
                            .Content("Build Bank...")
                            .OnClick(async () => await BuildBankAsync()),
                        new Button()
                            .Content("Choose Image && Detect")
                            .OnClick(async () => await DetectImage())
                    ),

                // Row 4: Info log
                new Label().Text("Info:"),
                new MultiLineTextBox()
                    .BindText(_vm.InfoLog)
                    .Height(80)
                    .FontFamily("Consolas"),

                // Row 5: Timing
                new TextBlock()
                    .BindText(_vm.TimingInfo)
                    .FontFamily("Consolas")
                    .FontSize(12)
            );
    }

    private Element BuildResultPanel()
    {
        _originalView = new Image().StretchMode(Stretch.Uniform).Width(DisplaySize).Height(DisplaySize);
        _heatmapView = new Image().StretchMode(Stretch.Uniform).Width(DisplaySize).Height(DisplaySize);
        _overlayView = new Image().StretchMode(Stretch.Uniform).Width(DisplaySize).Height(DisplaySize);
        _maskView = new Image().StretchMode(Stretch.Uniform).Width(DisplaySize).Height(DisplaySize);

        return new StackPanel()
            .Vertical()
            .Spacing(8)
            .Children(
                new Label().Text("Results:").Bold().FontSize(14),
                new UniformGrid()
                    .Columns(4)
                    .Rows(1)
                    .Spacing(8)
                    .Children(
                        ImageBox("Original", _originalView),
                        ImageBox("Anomaly Heatmap", _heatmapView),
                        ImageBox("Overlay", _overlayView),
                        ImageBox("Defect Mask", _maskView)
                    )
            );
    }

    private static Element ImageBox(string title, Image img)
    {
        return new StackPanel()
            .Vertical()
            .Spacing(4)
            .Center()
            .Children(
                new Label().Text(title).Center(),
                new Border()
                    .BorderThickness(1)
                    .Child(img)
            );
    }

    private void TryInitModel()
    {
        string path = _vm.ModelPath.Value;

        if (File.Exists(path))
        {
            try
            {
                _inference?.Dispose();
                string device = _vm.DeviceIndex.Value == 0 ? "CPU" : "GPU";
                _inference = new RadInference(path, device);
                _vm.AppendLog($"Model loaded ({_inference.ActiveDevice}): {Path.GetFileName(path)}");
            }
            catch (Exception ex)
            {
                _vm.AppendLog($"Failed to load model: {ex.Message}");
            }
        }
        else
        {
            _vm.AppendLog($"Model not found: {path}");
        }
    }

    private void BrowseModel()
    {
        if (_window == null) return;
        string? file = FileDialog.OpenFile(new OpenFileDialogOptions
        {
            Owner = _window.Handle,
            Title = "Select ONNX Model",
            Filter = "ONNX Files (*.onnx)|*.onnx|All Files (*.*)|*.*"
        });

        if (file != null && File.Exists(file))
        {
            _vm.ModelPath.Value = file;
            LoadModel(file);
        }
    }

    private void LoadModel(string path)
    {
        _inference?.Dispose();
        string device = _vm.DeviceIndex.Value == 0 ? "GPU" : "CPU";
        _inference = new RadInference(path, device);
        _vm.AppendLog($"Model loaded ({_inference.ActiveDevice}): {Path.GetFileName(path)}");
    }

    private async Task BuildBankAsync()
    {
        if (_window == null || _inference == null) return;

        string? folder = FileDialog.SelectFolder(new FolderDialogOptions
        {
            Owner = _window.Handle,
            Title = "Select folder with normal images"
        });

        if (folder == null) return;

        _vm.IsBusy.Value = true;
        _vm.AppendLog($"Building bank from: {folder}");

        try
        {
            string bankDir = Path.Combine(Path.GetDirectoryName(_vm.ModelPath.Value) ?? ".", "bank");
            await Task.Run(() =>
            {
                _inference.BuildBank(folder, bankDir, msg =>
                    Application.Current?.Dispatcher.Invoke(() => _vm.AppendLog(msg)));
            });

            _bankLoaded = true;
            _vm.IsBankLoaded.Value = true;
            _vm.AppendLog("Bank build complete.");
        }
        catch (Exception ex)
        {
            _vm.AppendLog($"Bank build failed: {ex.Message}");
        }
        finally
        {
            _vm.IsBusy.Value = false;
        }
    }

    private async Task DetectImage()
    {
        if (_window == null || _inference == null || !_bankLoaded) return;

        string? file = FileDialog.OpenFile(new OpenFileDialogOptions
        {
            Owner = _window.Handle,
            Title = "Select image for anomaly detection",
            Filter = "Images (*.png;*.jpg;*.jpeg;*.bmp)|*.png;*.jpg;*.jpeg;*.bmp|All Files (*.*)|*.*"
        });

        if (file == null) return;

        _vm.IsBusy.Value = true;
        _vm.AppendLog($"Detecting: {Path.GetFileName(file)}");

        try
        {
            float threshold = _vm.Threshold.Value;
            if (threshold <= 0) threshold = 0.15f;
            _inference.KImage = _vm.KImage.Value;

            var totalSw = System.Diagnostics.Stopwatch.StartNew();

            var (anomalyMap, score) = await Task.Run(() =>
                _inference.Detect(file, threshold, out _preMs, out _boneMs, out _postMs));

            var vizSw = System.Diagnostics.Stopwatch.StartNew();

            using var originalBitmap = ImagePreprocessor.LoadForDisplay(file);
            using var heatmapBitmap = VisualizationHelper.CreateHeatmap(anomalyMap, originalBitmap.Width, originalBitmap.Height);
            using var overlayBitmap = VisualizationHelper.CreateOverlay(originalBitmap, heatmapBitmap, 0.35f);
            using var maskBitmap = VisualizationHelper.CreateBinaryMask(anomalyMap, threshold, originalBitmap.Width, originalBitmap.Height);

            using var origRgba = originalBitmap.CloneAs<Rgba32>();
            using var origDisplay = VisualizationHelper.ResizeForDisplay(origRgba, DisplaySize);
            using var heatDisplay = VisualizationHelper.ResizeForDisplay(heatmapBitmap, DisplaySize);
            using var overDisplay = VisualizationHelper.ResizeForDisplay(overlayBitmap, DisplaySize);
            using var maskDisplay = VisualizationHelper.ResizeForDisplay(maskBitmap, DisplaySize);

            double vizMs = vizSw.Elapsed.TotalMilliseconds;
            double totalMs = totalSw.Elapsed.TotalMilliseconds;

            byte[] origPng = VisualizationHelper.ToPngBytes(origDisplay);
            byte[] heatPng = VisualizationHelper.ToPngBytes(heatDisplay);
            byte[] overPng = VisualizationHelper.ToPngBytes(overDisplay);
            byte[] maskPng = VisualizationHelper.ToPngBytes(maskDisplay);

            Application.Current?.Dispatcher.Invoke(() =>
            {
                if (_originalView != null) _originalView.Source = ImageSource.FromBytes(origPng);
                if (_heatmapView != null) _heatmapView.Source = ImageSource.FromBytes(heatPng);
                if (_overlayView != null) _overlayView.Source = ImageSource.FromBytes(overPng);
                if (_maskView != null) _maskView.Source = ImageSource.FromBytes(maskPng);
            });

            _vm.SetTiming(_preMs, _boneMs, _postMs, vizMs, totalMs);
            _vm.AppendLog($"Detection complete. Score: {score:F4}");
        }
        catch (Exception ex)
        {
            _vm.AppendLog($"[FATAL] {ex.GetType().Name}: {ex.Message}");
        }
        finally
        {
            _vm.IsBusy.Value = false;
        }
    }

}
