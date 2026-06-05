using System.Drawing;
using System.Windows.Forms;

partial class RADForm
{
    private System.ComponentModel.IContainer components = null;
    private Button btnSelectModel;
    private TextBox txtModelPath;
    private Button btnSelectBank;
    private TextBox txtBankDir;
    private Button btnBuildBank;
    private Label lblStatus;
    private ProgressBar progressBar;
    private ComboBox cmbDevice;

    protected override void Dispose(bool disposing)
    {
        if (disposing && (components != null))
        {
            components.Dispose();
        }
        base.Dispose(disposing);
    }

    private void InitializeComponent()
    {
        btnSelectModel = new Button();
        txtModelPath = new TextBox();
        btnSelectBank = new Button();
        txtBankDir = new TextBox();
        btnBuildBank = new Button();
        lblStatus = new Label();
        progressBar = new ProgressBar();
        cmbDevice = new ComboBox();
        button1 = new Button();
        SuspendLayout();
        // 
        // btnSelectModel
        // 
        btnSelectModel.Location = new Point(20, 20);
        btnSelectModel.Name = "btnSelectModel";
        btnSelectModel.Size = new Size(120, 30);
        btnSelectModel.TabIndex = 0;
        btnSelectModel.Text = "选择 ONNX 模型";
        // 
        // txtModelPath
        // 
        txtModelPath.Location = new Point(150, 23);
        txtModelPath.Name = "txtModelPath";
        txtModelPath.ReadOnly = true;
        txtModelPath.Size = new Size(400, 23);
        txtModelPath.TabIndex = 1;
        txtModelPath.Text = "Model/dinov3_multilayer.onnx";
        // 
        // btnSelectBank
        // 
        btnSelectBank.Location = new Point(20, 60);
        btnSelectBank.Name = "btnSelectBank";
        btnSelectBank.Size = new Size(120, 30);
        btnSelectBank.TabIndex = 2;
        btnSelectBank.Text = "选择记忆库目录";
        btnSelectBank.Visible = false;
        // 
        // txtBankDir
        // 
        txtBankDir.Location = new Point(150, 63);
        txtBankDir.Name = "txtBankDir";
        txtBankDir.ReadOnly = true;
        txtBankDir.Size = new Size(400, 23);
        txtBankDir.TabIndex = 3;
        txtBankDir.Visible = false;
        // 
        // btnBuildBank
        // 
        btnBuildBank.Enabled = false;
        btnBuildBank.Location = new Point(566, 19);
        btnBuildBank.Name = "btnBuildBank";
        btnBuildBank.Size = new Size(100, 30);
        btnBuildBank.TabIndex = 4;
        btnBuildBank.Text = "构建记忆库";
        // 
        // lblStatus
        // 
        lblStatus.Location = new Point(20, 109);
        lblStatus.Name = "lblStatus";
        lblStatus.Size = new Size(600, 25);
        lblStatus.TabIndex = 8;
        lblStatus.Text = "就绪";
        // 
        // progressBar
        // 
        progressBar.Location = new Point(20, 137);
        progressBar.Name = "progressBar";
        progressBar.Size = new Size(660, 20);
        progressBar.TabIndex = 10;
        // 
        // cmbDevice
        // 
        cmbDevice.DropDownStyle = ComboBoxStyle.DropDownList;
        cmbDevice.Items.AddRange(new object[] { "CPU", "GPU" });
        cmbDevice.Location = new Point(20, 167);
        cmbDevice.Name = "cmbDevice";
        cmbDevice.Size = new Size(100, 25);
        cmbDevice.TabIndex = 11;
        // 
        // button1
        // 
        button1.Location = new Point(684, 19);
        button1.Name = "button1";
        button1.Size = new Size(99, 30);
        button1.TabIndex = 14;
        button1.Text = "异常检测";
        button1.UseVisualStyleBackColor = true;
        button1.Click += button1_Click;
        // 
        // RADForm
        // 
        ClientSize = new Size(812, 280);
        Controls.Add(button1);
        Controls.Add(btnSelectModel);
        Controls.Add(txtModelPath);
        Controls.Add(btnSelectBank);
        Controls.Add(txtBankDir);
        Controls.Add(btnBuildBank);
        Controls.Add(lblStatus);
        Controls.Add(progressBar);
        Controls.Add(cmbDevice);
        Name = "RADForm";
        Text = "RAD 异常检测工具";
        ResumeLayout(false);
        PerformLayout();
    }

    private Button button1;
}