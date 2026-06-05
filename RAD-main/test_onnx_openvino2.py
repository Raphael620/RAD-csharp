import numpy as np
import os
import struct
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import sys
import time

# 兼容新旧 OpenVINO 导入
try:
    from openvino.runtime import Core, Tensor
except ImportError:
    from openvino import Core, Tensor

# -------------------------- 二进制记忆库读取 --------------------------
def read_2d_array_from_bin(file_path):
    with open(file_path, 'rb') as f:
        ndim = struct.unpack('i', f.read(4))[0]
        if ndim != 2:
            raise ValueError(f"期望二维数组，实际维度 {ndim}")
        rows = struct.unpack('i', f.read(4))[0]
        cols = struct.unpack('i', f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(rows, cols)
    print(f"读取 {file_path}: 形状 {data.shape}")
    return data

def read_3d_array_from_bin(file_path):
    with open(file_path, 'rb') as f:
        ndim = struct.unpack('i', f.read(4))[0]
        if ndim != 3:
            raise ValueError(f"期望三维数组，实际维度 {ndim}")
        d1 = struct.unpack('i', f.read(4))[0]
        d2 = struct.unpack('i', f.read(4))[0]
        d3 = struct.unpack('i', f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(d1, d2, d3)
    print(f"读取 {file_path}: 形状 {data.shape}")
    return data

def read_layers_from_bin(file_path):
    with open(file_path, 'rb') as f:
        count = struct.unpack('i', f.read(4))[0]
        layers = []
        for _ in range(count):
            layers.append(struct.unpack('i', f.read(4))[0])
    print(f"读取 {file_path}: 层索引 {layers}")
    return layers

# -------------------------- 图像预处理 --------------------------
def preprocess_image(image_path, input_size=448):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.CenterCrop(input_size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
    img_np = img_tensor.numpy()
    img_np = np.expand_dims(img_np, axis=0).astype(np.float32)
    return img_np

# -------------------------- 工具函数 --------------------------
def normalize(vec):
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm

def resize_bilinear(src, dst_h, dst_w):
    src_h, src_w = src.shape
    scale_x = (src_w - 1) / (dst_w - 1) if dst_w > 1 else 0
    scale_y = (src_h - 1) / (dst_h - 1) if dst_h > 1 else 0
    dst = np.zeros((dst_h, dst_w), dtype=np.float32)
    for y in range(dst_h):
        fy = y * scale_y
        y0 = int(fy)
        y1 = min(y0 + 1, src_h - 1)
        wy1 = fy - y0
        wy0 = 1 - wy1
        for x in range(dst_w):
            fx = x * scale_x
            x0 = int(fx)
            x1 = min(x0 + 1, src_w - 1)
            wx1 = fx - x0
            wx0 = 1 - wx1
            v00 = src[y0, x0]
            v01 = src[y0, x1]
            v10 = src[y1, x0]
            v11 = src[y1, x1]
            v0 = v00 * wx0 + v01 * wx1
            v1 = v10 * wx0 + v11 * wx1
            dst[y, x] = v0 * wy0 + v1 * wy1
    return dst

def gaussian_kernel(size=5, sigma=1.0):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    kernel = kernel / np.sum(kernel)
    return kernel

def apply_gaussian_blur(anomaly_map, kernel_size=5, sigma=1.0):
    from scipy.ndimage import convolve
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve(anomaly_map, kernel, mode='reflect')

# -------------------------- 主推理类 --------------------------
class RADInference:
    def __init__(self, onnx_path, bank_dir, k_image=150, layer_weights=None):
        print("初始化 RADInference...")
        self.k_image = k_image

        # 加载 ONNX 模型
        print(f"加载 ONNX 模型: {onnx_path}")
        self.core = Core()
        model = self.core.read_model(onnx_path)
        self.compiled_model = self.core.compile_model(model, "CPU")
        self.infer_request = self.compiled_model.create_infer_request()
        print("模型加载成功，输出名称列表：")
        for out in self.compiled_model.outputs:
            print(f"  {out.any_name}")

        # 加载记忆库
        print(f"加载记忆库: {bank_dir}")
        layers_path = os.path.join(bank_dir, "layers.bin")
        self.layers = read_layers_from_bin(layers_path)
        self.num_layers = len(self.layers)
        self.cls_banks = []
        self.patch_banks = []
        for layer in self.layers:
            cls_path = os.path.join(bank_dir, f"cls_layer{layer}.bin")
            patch_path = os.path.join(bank_dir, f"patch_layer{layer}.bin")
            self.cls_banks.append(read_2d_array_from_bin(cls_path))      # [N, C]
            self.patch_banks.append(read_3d_array_from_bin(patch_path))  # [N, L, C]

        # 预归一化 patch 记忆库（加速后续计算）
        print("预归一化 patch 记忆库...")
        self.patch_banks_norm = []
        for pb in self.patch_banks:
            norms = np.linalg.norm(pb, axis=2, keepdims=True) + 1e-8
            pb_norm = pb / norms
            self.patch_banks_norm.append(pb_norm)

        # 层权重
        if layer_weights is None:
            self.layer_weights = np.ones(self.num_layers) / self.num_layers
        else:
            self.layer_weights = np.array(layer_weights) / sum(layer_weights)

        # 其他常量
        self.embed_dim = self.cls_banks[0].shape[1]      # 384
        self.num_patches = self.patch_banks[0].shape[1]  # 784
        self.patch_h = self.patch_w = int(np.sqrt(self.num_patches))  # 28
        print(f"初始化完成: 层数={self.num_layers}, 嵌入维度={self.embed_dim}, patch数={self.num_patches}")

    def infer(self, image_path, output_size=256, apply_smoothing=True):
        print("\n===== 开始推理 =====")
        start_total = time.time()

        # 1. 预处理
        print("步骤1: 图像预处理...")
        t = time.time()
        input_data = preprocess_image(image_path)
        print(f"  耗时: {time.time()-t:.2f}s")

        # 2. 创建 Tensor 并设置输入
        print("步骤2: 创建输入 Tensor...")
        t = time.time()
        input_tensor = Tensor(input_data)
        self.infer_request.set_input_tensor(0, input_tensor)
        print(f"  耗时: {time.time()-t:.2f}s")

        # 3. 推理
        print("步骤3: 执行推理...")
        t = time.time()
        self.infer_request.infer()
        print(f"  耗时: {time.time()-t:.2f}s")

        # 4. 获取各层输出
        print("步骤4: 获取输出张量...")
        t = time.time()
        patch_list = []
        cls_list = []
        for idx in self.layers:
            patch_key = f'patch_{idx}'
            cls_key = f'cls_{idx}'
            patch_tensor = self.infer_request.get_tensor(patch_key)
            cls_tensor = self.infer_request.get_tensor(cls_key)
            patch_list.append(patch_tensor.data)  # [1, L, C]
            cls_list.append(cls_tensor.data)      # [1, C]
        print(f"  耗时: {time.time()-t:.2f}s")

        # 5. 图像级检索（使用最高层 CLS）
        print("步骤5: 图像级检索...")
        t = time.time()
        query_cls = cls_list[-1][0]                # [C]
        norm_query_cls = normalize(query_cls)
        cls_bank_global = self.cls_banks[-1]       # [N, C]
        # 计算余弦相似度
        cls_norms = np.linalg.norm(cls_bank_global, axis=1) + 1e-8
        similarities = np.dot(cls_bank_global, norm_query_cls) / cls_norms
        topk_indices = np.argsort(similarities)[-self.k_image:][::-1]
        print(f"  最高层检索完成，top-{self.k_image} 索引: {topk_indices[:5]}...")
        print(f"  耗时: {time.time()-t:.2f}s")

        # 6. 逐层 patch-KNN（优化版本）
        print("步骤6: 逐层 patch-KNN...")
        fused_scores = np.zeros(self.num_patches)
        for li in range(self.num_layers):
            layer_start = time.time()
            print(f"  处理层 {self.layers[li]}...")
            test_patch = patch_list[li][0]  # [L, C]

            # 归一化测试 patch
            test_patch_norm = np.zeros_like(test_patch)
            for p in range(self.num_patches):
                test_patch_norm[p] = normalize(test_patch[p])

            # 从预归一化的记忆库中取出 top-k 个训练样本的 patch 特征 [k, L, C]
            bank_patch_norm = self.patch_banks_norm[li][topk_indices]  # [k, L, C]

            # 重塑为 [k*L, C]
            # kL = self.k_image * self.num_patches
            # bank_flat = bank_patch_norm.reshape(kL, self.embed_dim)  # [117600, 384]
            
            k_actual = len(topk_indices)
            kL_actual = k_actual * self.num_patches
            bank_flat = bank_patch_norm.reshape(kL_actual, self.embed_dim)

            # 对每个查询 patch 计算与所有 bank patch 的相似度
            layer_scores = np.zeros(self.num_patches)
            for p in range(self.num_patches):
                q = test_patch_norm[p]  # [384]
                sims = np.dot(bank_flat, q)  # [kL]
                max_sim = np.max(sims)
                layer_scores[p] = 1.0 - max_sim

                # 可选：每 100 个打印一次进度
                if (p + 1) % 200 == 0:
                    print(f"    已处理 {p+1}/{self.num_patches} patches")

            fused_scores += self.layer_weights[li] * layer_scores
            print(f"    层 {self.layers[li]} 处理完成，分数范围 [{layer_scores.min():.4f}, {layer_scores.max():.4f}], 耗时 {time.time()-layer_start:.2f}s")

        # 7. 重塑为 28x28
        patch_map = fused_scores.reshape(self.patch_h, self.patch_w)

        # 8. 上采样
        print("步骤8: 上采样...")
        t = time.time()
        anomaly_map = resize_bilinear(patch_map, output_size, output_size)
        print(f"  耗时: {time.time()-t:.2f}s")

        # 9. 高斯平滑
        if apply_smoothing:
            print("步骤9: 高斯平滑...")
            t = time.time()
            anomaly_map = apply_gaussian_blur(anomaly_map, kernel_size=5, sigma=1.0)
            print(f"  耗时: {time.time()-t:.2f}s")

        # 10. 图像级分数
        img_score = np.max(anomaly_map)
        print(f"步骤10: 图像级分数 = {img_score:.4f}")
        print(f"===== 推理结束，总耗时 {time.time()-start_total:.2f}s =====\n")

        return anomaly_map, img_score

    def release(self):
        pass

# -------------------------- 可视化 --------------------------
def visualize(image_path, anomaly_map, img_score, save_path=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 归一化异常图
    am_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    am_uint8 = (am_norm * 255).astype(np.uint8)

    # 生成热图
    heatmap = cv2.applyColorMap(am_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 调整大小
    h, w = img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # 叠加
    alpha = 0.5
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_resized, alpha, 0)

    # 显示
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(heatmap_resized)
    plt.title("Anomaly Heatmap")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(overlay)
    plt.title(f"Overlay (score={img_score:.4f})")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(am_norm, cmap='jet')
    plt.title("Raw Anomaly Map")
    plt.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存至 {save_path}")
    plt.show()

# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    # 参数配置
    onnx_file = "dinov3_multilayer.onnx"
    bank_directory = "bin"
    test_image = "000.png"
    output_vis = "result.png"

    # 检查文件是否存在
    if not os.path.exists(onnx_file):
        print(f"错误: ONNX 文件不存在: {onnx_file}")
        sys.exit(1)
    if not os.path.exists(bank_directory):
        print(f"错误: 记忆库目录不存在: {bank_directory}")
        sys.exit(1)
    if not os.path.exists(test_image):
        print(f"错误: 测试图像不存在: {test_image}")
        sys.exit(1)

    try:
        # 初始化
        rad = RADInference(onnx_file, bank_directory, k_image=150)

        # 推理
        anomaly_map, score = rad.infer(test_image, output_size=256, apply_smoothing=True)

        # 可视化
        visualize(test_image, anomaly_map, score, save_path=output_vis)

        print("程序正常结束。")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()