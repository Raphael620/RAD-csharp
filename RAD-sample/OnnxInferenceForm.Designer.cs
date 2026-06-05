using System.Windows.Forms;
using System.Drawing;

namespace RAD_WinForms
{
    partial class OnnxInferenceForm
    {
        /// <summary>
        /// 必需的设计器变量。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 清理所有正在使用的资源。
        /// </summary>
        /// <param name="disposing">如果应释放托管资源，为 true；否则为 false。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows 窗体设计器生成的代码

        /// <summary>
        /// 设计器支持所需的方法 - 不要修改
        /// 使用代码编辑器修改此方法的内容。
        /// </summary>
        private void InitializeComponent()
        {
            btnLoadImage = new Button();
            btnDetect = new Button();
            btnBatch = new Button();
            lblResult = new Label();
            pictureBoxOriginal = new PictureBox();
            pictureBoxHeatmap = new PictureBox();
            pictureBoxOverlay = new PictureBox();
            pictureBoxRaw = new PictureBox();
            openFileDialog = new OpenFileDialog();
            folderBrowserDialog = new FolderBrowserDialog();
            progressBar = new ProgressBar();
            lblProgress = new Label();
            lblOrig = new Label();
            lblHeat = new Label();
            lblOver = new Label();
            lblRaw = new Label();
            lblTime = new Label();
            decimalUpDown1 = new CustomControls.DecimalUpDown();
            lblThreshold = new Label();
            lblOKNG = new Label();
            ((System.ComponentModel.ISupportInitialize)pictureBoxOriginal).BeginInit();
            ((System.ComponentModel.ISupportInitialize)pictureBoxHeatmap).BeginInit();
            ((System.ComponentModel.ISupportInitialize)pictureBoxOverlay).BeginInit();
            ((System.ComponentModel.ISupportInitialize)pictureBoxRaw).BeginInit();
            ((System.ComponentModel.ISupportInitialize)decimalUpDown1).BeginInit();
            SuspendLayout();
            // 
            // btnLoadImage
            // 
            btnLoadImage.Location = new Point(20, 20);
            btnLoadImage.Name = "btnLoadImage";
            btnLoadImage.Size = new Size(100, 30);
            btnLoadImage.TabIndex = 0;
            btnLoadImage.Text = "选择图像";
            btnLoadImage.Click += BtnLoadImage_Click;
            // 
            // btnDetect
            // 
            btnDetect.Enabled = false;
            btnDetect.Location = new Point(157, 20);
            btnDetect.Name = "btnDetect";
            btnDetect.Size = new Size(100, 30);
            btnDetect.TabIndex = 1;
            btnDetect.Text = "检测";
            btnDetect.Click += BtnDetect_Click;
            // 
            // btnBatch
            // 
            btnBatch.Enabled = false;
            btnBatch.Location = new Point(300, 20);
            btnBatch.Name = "btnBatch";
            btnBatch.Size = new Size(100, 30);
            btnBatch.TabIndex = 15;
            btnBatch.Text = "批量推理";
            btnBatch.Click += BtnBatch_Click;
            // 
            // lblResult
            // 
            lblResult.Location = new Point(413, 25);
            lblResult.Name = "lblResult";
            lblResult.Size = new Size(200, 20);
            lblResult.TabIndex = 2;
            lblResult.Text = "就绪";
            // 
            // pictureBoxOriginal
            // 
            pictureBoxOriginal.BorderStyle = BorderStyle.FixedSingle;
            pictureBoxOriginal.Location = new Point(20, 70);
            pictureBoxOriginal.Name = "pictureBoxOriginal";
            pictureBoxOriginal.Size = new Size(256, 256);
            pictureBoxOriginal.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBoxOriginal.TabIndex = 3;
            pictureBoxOriginal.TabStop = false;
            // 
            // pictureBoxHeatmap
            // 
            pictureBoxHeatmap.BorderStyle = BorderStyle.FixedSingle;
            pictureBoxHeatmap.Location = new Point(300, 70);
            pictureBoxHeatmap.Name = "pictureBoxHeatmap";
            pictureBoxHeatmap.Size = new Size(256, 256);
            pictureBoxHeatmap.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBoxHeatmap.TabIndex = 4;
            pictureBoxHeatmap.TabStop = false;
            // 
            // pictureBoxOverlay
            // 
            pictureBoxOverlay.BorderStyle = BorderStyle.FixedSingle;
            pictureBoxOverlay.Location = new Point(580, 70);
            pictureBoxOverlay.Name = "pictureBoxOverlay";
            pictureBoxOverlay.Size = new Size(256, 256);
            pictureBoxOverlay.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBoxOverlay.TabIndex = 5;
            pictureBoxOverlay.TabStop = false;
            // 
            // pictureBoxRaw
            // 
            pictureBoxRaw.BorderStyle = BorderStyle.FixedSingle;
            pictureBoxRaw.Location = new Point(860, 70);
            pictureBoxRaw.Name = "pictureBoxRaw";
            pictureBoxRaw.Size = new Size(256, 256);
            pictureBoxRaw.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBoxRaw.TabIndex = 6;
            pictureBoxRaw.TabStop = false;
            // 
            // openFileDialog
            // 
            openFileDialog.Filter = "图像文件|*.png;*.jpg;*.jpeg;*.bmp";
            // 
            // folderBrowserDialog
            // 
            folderBrowserDialog.Description = "选择图片文件夹";
            // 
            // progressBar
            // 
            progressBar.Location = new Point(20, 340);
            progressBar.Name = "progressBar";
            progressBar.Size = new Size(500, 23);
            progressBar.TabIndex = 16;
            // 
            // lblProgress
            // 
            lblProgress.Location = new Point(20, 370);
            lblProgress.Name = "lblProgress";
            lblProgress.Size = new Size(500, 20);
            lblProgress.TabIndex = 17;
            // 
            // lblOrig
            // 
            lblOrig.Location = new Point(0, 0);
            lblOrig.Name = "lblOrig";
            lblOrig.Size = new Size(100, 23);
            lblOrig.TabIndex = 7;
            // 
            // lblHeat
            // 
            lblHeat.Location = new Point(0, 0);
            lblHeat.Name = "lblHeat";
            lblHeat.Size = new Size(100, 23);
            lblHeat.TabIndex = 8;
            // 
            // lblOver
            // 
            lblOver.Location = new Point(0, 0);
            lblOver.Name = "lblOver";
            lblOver.Size = new Size(100, 23);
            lblOver.TabIndex = 9;
            // 
            // lblRaw
            // 
            lblRaw.Location = new Point(0, 0);
            lblRaw.Name = "lblRaw";
            lblRaw.Size = new Size(100, 23);
            lblRaw.TabIndex = 10;
            // 
            // lblTime
            // 
            lblTime.Location = new Point(580, 340);
            lblTime.Name = "lblTime";
            lblTime.Size = new Size(300, 20);
            lblTime.TabIndex = 11;
            // 
            // decimalUpDown1
            // 
            decimalUpDown1.AutoSelectOnValueChanged = false;
            decimalUpDown1.DecimalPlaces = 3;
            decimalUpDown1.Increment = new decimal(new int[] { 1, 0, 0, 131072 });
            decimalUpDown1.Location = new Point(668, 24);
            decimalUpDown1.MinimumSize = new Size(50, 0);
            decimalUpDown1.Name = "decimalUpDown1";
            decimalUpDown1.Size = new Size(98, 23);
            decimalUpDown1.TabIndex = 12;
            decimalUpDown1.TextAlign = HorizontalAlignment.Right;
            decimalUpDown1.Value = new decimal(new int[] { 105, 0, 0, 196608 });
            // 
            // lblThreshold
            // 
            lblThreshold.AutoSize = true;
            lblThreshold.Location = new Point(630, 26);
            lblThreshold.Name = "lblThreshold";
            lblThreshold.Size = new Size(32, 17);
            lblThreshold.TabIndex = 13;
            lblThreshold.Text = "阈值";
            // 
            // lblOKNG
            // 
            lblOKNG.AutoSize = true;
            lblOKNG.Font = new Font("Microsoft YaHei UI", 15.75F, FontStyle.Regular, GraphicsUnit.Point, 134);
            lblOKNG.Location = new Point(780, 19);
            lblOKNG.Name = "lblOKNG";
            lblOKNG.Size = new Size(0, 28);
            lblOKNG.TabIndex = 14;
            // 
            // TestForm2
            // 
            ClientSize = new Size(1184, 410);
            Controls.Add(lblProgress);
            Controls.Add(progressBar);
            Controls.Add(btnBatch);
            Controls.Add(lblOKNG);
            Controls.Add(lblThreshold);
            Controls.Add(decimalUpDown1);
            Controls.Add(lblTime);
            Controls.Add(btnLoadImage);
            Controls.Add(btnDetect);
            Controls.Add(lblResult);
            Controls.Add(pictureBoxOriginal);
            Controls.Add(pictureBoxHeatmap);
            Controls.Add(pictureBoxOverlay);
            Controls.Add(pictureBoxRaw);
            Controls.Add(lblOrig);
            Controls.Add(lblHeat);
            Controls.Add(lblOver);
            Controls.Add(lblRaw);
            Name = "TestForm2";
            Text = "RAD 异常检测";
            ((System.ComponentModel.ISupportInitialize)pictureBoxOriginal).EndInit();
            ((System.ComponentModel.ISupportInitialize)pictureBoxHeatmap).EndInit();
            ((System.ComponentModel.ISupportInitialize)pictureBoxOverlay).EndInit();
            ((System.ComponentModel.ISupportInitialize)pictureBoxRaw).EndInit();
            ((System.ComponentModel.ISupportInitialize)decimalUpDown1).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Button btnLoadImage;
        private Button btnDetect;
        private Label lblResult;
        private PictureBox pictureBoxOriginal;
        private PictureBox pictureBoxHeatmap;
        private PictureBox pictureBoxOverlay;
        private PictureBox pictureBoxRaw;
        private OpenFileDialog openFileDialog;
        private Label lblOrig;
        private Label lblHeat;
        private Label lblOver;
        private Label lblRaw;
        private Label lblTime;
        private CustomControls.DecimalUpDown decimalUpDown1;
        private Label lblThreshold;
        private Label lblOKNG;
        private Button btnBatch;
        private FolderBrowserDialog folderBrowserDialog;
        private ProgressBar progressBar;
        private Label lblProgress;
    }
}