using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using System.Threading.Tasks;
using System.Diagnostics;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using RAD_WinForms;          // 引入 RADInference 所在命名空间

public partial class RADForm : Form
{
    private RADInference rad;      // 改用 RADInference
    private string bankDir;

    public RADForm()
    {
        InitializeComponent();
        this.Text = "RAD 异常检测工具";

        // 确保进度条初始样式为连续
        progressBar.Style = ProgressBarStyle.Continuous;
        progressBar.Minimum = 0;
        progressBar.Maximum = 100;
        progressBar.Value = 0;
        UpdateBuildButton();
        // 绑定事件
        btnSelectModel.Click += (s, e) => SelectFile("选择 ONNX 模型", "onnx", path => { txtModelPath.Text = path; UpdateBuildButton(); });
        btnBuildBank.Click += async (s, e) => await BuildBank();
        cmbDevice.SelectedIndex = 1;   // 设置默认设备
    }

    private void SelectFile(string title, string filter, Action<string> onSelected)
    {
        using var ofd = new OpenFileDialog();
        ofd.Title = title;
        ofd.Filter = $"文件|*.{filter.Replace("|", ";*.")}";
        if (ofd.ShowDialog() == DialogResult.OK)
            onSelected(ofd.FileName);
    }

    private void SelectFolder(string title, Action<string> onSelected)
    {
        using var fbd = new FolderBrowserDialog();
        fbd.Description = title;
        if (fbd.ShowDialog() == DialogResult.OK)
            onSelected(fbd.SelectedPath);
    }

    private void UpdateBuildButton()
    {
        btnBuildBank.Enabled = !string.IsNullOrEmpty(txtModelPath.Text);
    }

    private async Task BuildBank()
    {
        // 获取 UI 控件值（在 UI 线程上完成）
        string modelPath = txtModelPath.Text;
        string device = cmbDevice.Text;       // ← 在 UI 线程读取，避免跨线程
        string outputDir = "./Bank";

        string imageFolder = null;
        using (var fbd = new FolderBrowserDialog())
        {
            fbd.Description = "选择包含正常图像的文件夹";
            if (fbd.ShowDialog() != DialogResult.OK) return;
            imageFolder = fbd.SelectedPath;
        }

        btnBuildBank.Enabled = false;
        progressBar.Visible = true;
        progressBar.Value = 0;
        lblStatus.Text = "正在构建记忆库...";

        // 进度报告（始终在 UI 线程执行）
        var progress = new Progress<string>(msg =>
        {
            lblStatus.Text = msg;
            int percent = ParseProgressFromMessage(msg);
            if (percent >= 0)
            {
                if (progressBar.Style != ProgressBarStyle.Continuous)
                    progressBar.Style = ProgressBarStyle.Continuous;
                progressBar.Value = Math.Min(percent, 100);
            }
            else
            {
                if (progressBar.Style != ProgressBarStyle.Marquee)
                {
                    progressBar.Style = ProgressBarStyle.Marquee;
                    progressBar.MarqueeAnimationSpeed = 30;
                }
            }
        });

        try
        {
            // 耗时操作放到后台线程，并传递所有需要的参数（不传递控件）
            await Task.Run(async () =>
            {
                using (var builder = new MemoryBankBuilder(modelPath, device))
                {
                    await builder.BuildAsync(imageFolder, outputDir, progress);
                }
            });
            MessageBox.Show("记忆库构建完成！", "提示");
        }
        catch (Exception ex)
        {
            MessageBox.Show($"构建失败: {ex.Message}", "错误");
        }
        finally
        {
            // 回到 UI 线程恢复控件状态
            btnBuildBank.Enabled = true;
            progressBar.Visible = false;
            progressBar.Style = ProgressBarStyle.Continuous;
        }
    }

    private int ParseProgressFromMessage(string msg)
    {
        if (string.IsNullOrEmpty(msg)) return -1;

        var match = System.Text.RegularExpressions.Regex.Match(msg, @"(\d+)\/(\d+)");
        if (match.Success && int.TryParse(match.Groups[1].Value, out int current) && int.TryParse(match.Groups[2].Value, out int total))
        {
            return (int)((double)current / total * 100);
        }

        match = System.Text.RegularExpressions.Regex.Match(msg, @"(\d+)%");
        if (match.Success && int.TryParse(match.Groups[1].Value, out int percent))
        {
            return percent;
        }

        return -1;
    }

    protected override void OnFormClosing(FormClosingEventArgs e)
    {
        rad?.Dispose();
        base.OnFormClosing(e);
    }

    private void button1_Click(object sender, EventArgs e)
    {
        OnnxInferenceForm testForm = new OnnxInferenceForm();
        testForm.ShowDialog();
    }
}