# RAD C# — 跨平台图像异常检测

基于 DINOv3 特征的桌面端异常检测应用。灵感来源于 [RAD](https://github.com/longkukuhi/RAD) 方法——无需训练，仅需少量正常样本构建记忆库，通过 patch 级 KNN 检索实现异常检测。

![主界面截图](sample/MainUI.jpg)

## 特性

- **无需训练** — 仅需几张正常图像即可构建记忆库
- **跨平台** — Windows / Linux / macOS 支持（MewUI + .NET NativeAOT）
- **GPU 加速** — Windows 下支持 DirectML，自动回退 CPU
- **单文件发布** — NativeAOT 编译为独立可执行文件

## 工作流程

1. **加载模型** — 选择 DINOv3 ONNX 模型文件
2. **构建记忆库** — 指定正常样本文件夹，提取多层特征
3. **异常检测** — 选择待检测图像，生成热力图、叠加图和二值掩码

## 快速开始

### 环境要求
- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)

### RAD.UI（桌面 GUI）
```bash
dotnet run --project src/RAD.UI/RAD.UI.csproj
```

### RAD.WebUI（Web 界面）
```bash
dotnet run --project src/RAD.WebUI/RAD.WebUI.csproj
# 浏览器打开 http://localhost:8080（可通过 -- <端口号> 修改端口）
```

**RAD.WebUI** 是一个零外部依赖的轻量 Web 界面，适合无桌面环境、ARM 开发板或远程访问：

- 品类管理，bank/test 图片独立存放
- 浏览器上传图片
- SSE 实时推送构建进度
- AJAX 检测结果在不刷新页面的情况下更新图片框
- 局域网内任意设备均可访问

### 发布（单文件可执行）
```bash
dotnet publish src/RAD.UI/RAD.UI.csproj -c Release -r win-x64 --self-contained
# 输出: src/RAD.UI/bin/Release/net10.0/win-x64/publish/RAD.UI.exe
```

将 `win-x64` 替换为 `linux-x64` 或 `osx-arm64` 即可。

## 项目结构

```
RAD-csharp/
├── model/                  # DINOv3 ONNX 模型
│   └── dinov3_multilayer.onnx
├── sample/                 # 示例图片
│   ├── OK/                 # 正常样本（用于构建记忆库）
│   └── NG/                 # 异常样本（用于测试）
├── src/
│   ├── RAD.Detector/       # 核心推理库
│   │   ├── ONNX Runtime 推理
│   │   ├── 记忆库构建与 KNN 检索
│   │   ├── 图像预处理（ImageSharp）
│   │   └── 可视化（热力图、叠加、掩码）
│   ├── RAD.UI/             # 桌面 GUI（MewUI）
│   └── RAD.WebUI/          # Web 界面（HttpListener，零依赖）
└── RAD-main/               # Python 参考代码
```

## 依赖

| 库 | 用途 |
|---------|---------|
| ONNX Runtime (DirectML) | DINOv3 模型推理 |
| SixLabors.ImageSharp | 图像预处理与可视化 |
| MewUI | 跨平台桌面 GUI |

## 参考

- [RAD 论文](https://github.com/longkukuhi/RAD)

---

[English](README.md)

<div align="center">
如果这个项目对你有帮助，请给一个 ⭐ Star 支持！
</div>
