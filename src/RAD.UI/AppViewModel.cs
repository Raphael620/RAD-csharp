using Aprillz.MewUI;

namespace RAD.UI;

/// <summary>
/// ViewModel for the main window. Uses ObservableValue for MewUI data binding.
/// </summary>
public class AppViewModel
{
    // Model path — just use current directory
    public ObservableValue<string> ModelPath { get; } = new(
        Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), "dinov3_multilayer.onnx")));

    // Device selection: 0 = CPU, 1 = GPU
    public ObservableValue<int> DeviceIndex { get; } = new(0);
    public string[] DeviceOptions { get; } = ["CPU", "GPU"];

    // kImage (top-K images for retrieval)
    public ObservableValue<int> KImage { get; } = new(10);

    // Threshold
    public ObservableValue<float> Threshold { get; } = new(0.12f);

    // Log / info area
    public ObservableValue<string> InfoLog { get; } = new("Ready.");

    // Timing info
    public ObservableValue<string> TimingInfo { get; } = new("Pre: --ms  Bone: --ms  Post: --ms  Viz: --ms  Total: --ms");

    // Status
    public ObservableValue<bool> IsBusy { get; } = new(false);
    public ObservableValue<bool> IsBankLoaded { get; } = new(false);

    // Image display sources (as byte arrays for MewUI ImageSource)
    public ObservableValue<byte[]?> OriginalImage { get; } = new(null as byte[]);
    public ObservableValue<byte[]?> HeatmapImage { get; } = new(null as byte[]);
    public ObservableValue<byte[]?> OverlayImage { get; } = new(null as byte[]);
    public ObservableValue<byte[]?> MaskImage { get; } = new(null as byte[]);

    public void AppendLog(string msg)
    {
        var current = InfoLog.Value;
        var time = DateTime.Now.ToString("HH:mm:ss");
        var log = $"[{time}] {msg}\n{current}";
        if (log.Length > 5000) log = log[..5000];
        InfoLog.Value = log;
    }

    public void SetTiming(double pre, double bone, double post, double viz, double total)
    {
        TimingInfo.Value = $"Pre: {pre:F1}ms  Bone: {bone:F1}ms  Post: {post:F1}ms  Viz: {viz:F1}ms  Total: {total:F1}ms";
    }

    public void ClearImages()
    {
        OriginalImage.Value = null;
        HeatmapImage.Value = null;
        OverlayImage.Value = null;
        MaskImage.Value = null;
    }
}
