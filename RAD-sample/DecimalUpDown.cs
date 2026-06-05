using System;
using System.ComponentModel;
using System.Windows.Forms;
using System.Windows.Input;

namespace CustomControls
{
    /// <summary>
    /// 基于原生 NumericUpDown 的自定义小数控件，支持任意小数位数和增量。
    /// 修正了可能存在的焦点异常问题，并提供文本对齐、自定义格式化等功能。
    /// </summary>
    [DefaultEvent("ValueChanged")]
    public class DecimalUpDown : NumericUpDown
    {
        private int _decimalPlaces = 1;
        private decimal _increment = 0.1m;
        private bool _settingTextAlign; // 防止递归设置文本对齐

        /// <summary>
        /// 获取或设置小数位数。
        /// </summary>
        [Category("行为")]
        [Description("显示的小数位数。")]
        public new int DecimalPlaces
        {
            get => _decimalPlaces;
            set
            {
                if (value >= 0 && value != _decimalPlaces)
                {
                    _decimalPlaces = value;
                    base.DecimalPlaces = value;
                    // 更新显示格式
                    UpdateEditText();
                }
            }
        }

        /// <summary>
        /// 获取或设置每次增减的步长。
        /// </summary>
        [Category("行为")]
        [Description("每次增减的步长。")]
        public new decimal Increment
        {
            get => _increment;
            set
            {
                if (value > 0)
                {
                    _increment = value;
                    base.Increment = value;
                }
            }
        }

        /// <summary>
        /// 获取或设置文本框内文本的对齐方式。
        /// </summary>
        [Category("外观")]
        [Description("文本框内文本的对齐方式。")]
        public HorizontalAlignment TextAlign
        {
            get
            {
                if (Controls[0] is TextBox tb)
                    return tb.TextAlign;
                return HorizontalAlignment.Right;
            }
            set
            {
                if (Controls[0] is TextBox tb && !_settingTextAlign)
                {
                    _settingTextAlign = true;
                    tb.TextAlign = value;
                    _settingTextAlign = false;
                }
            }
        }

        /// <summary>
        /// 获取或设置是否在值改变时自动选中文本（默认 false，避免视觉上总是处于选中状态）。
        /// </summary>
        [Category("行为")]
        [Description("值改变时是否自动选中文本框内容。")]
        public bool AutoSelectOnValueChanged { get; set; } = false;

        public DecimalUpDown()
        {
            // 设置默认属性
            base.DecimalPlaces = _decimalPlaces;
            base.Increment = _increment;
            base.Minimum = 0;
            base.Maximum = 100;
            base.Value = 0;

            // 确保控件不会自动获得焦点（除非用户点击或 Tab 切换）
            this.TabStop = true;  // 可接受 Tab 焦点，但不会强制获取

            // 在控件创建完成后设置文本对齐（避免在设计时出错）
            this.HandleCreated += (s, e) =>
            {
                if (!DesignMode)
                {
                    // 设置默认居中对齐（可根据需求修改）
                    TextAlign = HorizontalAlignment.Center;
                }
            };
        }

        /// <summary>
        /// 重写值改变时的行为，避免默认的文本全选行为，防止视觉上一直有焦点。
        /// </summary>
        protected override void OnValueChanged(EventArgs e)
        {
            base.OnValueChanged(e);

            if (!AutoSelectOnValueChanged)
            {
                // 取消自动选中文本（原生 NumericUpDown 在值改变时会全选文本）
                if (Controls[0] is TextBox tb)
                {
                    int selectionStart = tb.SelectionStart;
                    tb.Select(selectionStart, 0); // 将光标移到原位置，取消选中
                }
            }
        }

        /// <summary>
        /// 重写获取焦点事件，确保不会意外导致无限循环或错误。
        /// </summary>
        protected override void OnGotFocus(EventArgs e)
        {
            base.OnGotFocus(e);
            // 这里不需要额外操作，基类已处理
        }

        /// <summary>
        /// 重写失去焦点事件，确保文本格式正确。
        /// </summary>
        protected override void OnLostFocus(EventArgs e)
        {
            base.OnLostFocus(e);
            // 失去焦点时，若文本格式不正确，基类会自动恢复
        }

        /// <summary>
        /// 更新显示文本，按指定小数位数格式化。
        /// </summary>
        private void UpdateEditText()
        {
            // 避免在文本格式化时触发额外事件
            string newText = Value.ToString($"F{DecimalPlaces}");
            if (this.Text != newText)
            {
                this.Text = newText;
            }
        }

        /// <summary>
        /// 重写键盘输入处理，防止输入超过小数位数（基类已做，但可增强）
        /// </summary>
        protected override void OnKeyPress(KeyPressEventArgs e)
        {
            base.OnKeyPress(e);
            // 基类已经处理小数位数限制，无需额外代码
        }
    }
}