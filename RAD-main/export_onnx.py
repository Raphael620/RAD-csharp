import torch
from dinov3.hub.backbones import load_dinov3_model

# 加载 DINOv3 模型（与 RAD 使用的相同）
model = load_dinov3_model(
    'dinov3_vits16',
    pretrained_weight_path='./weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
)
model.eval()

# 包装模型，使其 forward 返回指定层的 patch tokens 和 cls token
class DINOv3Wrapper(torch.nn.Module):
    def __init__(self, model, layer_indices):
        super().__init__()
        self.model = model
        self.layer_indices = layer_indices

    def forward(self, x):
        # x: [N, C, H, W]
        intermediates = self.model.get_intermediate_layers(
            x,
            n=self.layer_indices,
            return_class_token=True,
            reshape=False,
            norm=True
        )
        patch_tokens = []
        cls_tokens = []
        for patch, cls in intermediates:
            patch_tokens.append(patch)  # [N, L, C]
            cls_tokens.append(cls)      # [N, C]
        # 将所有层的输出拼接成一个大的输出（或返回 tuple）
        # 这里返回两个列表，但 ONNX 要求固定数量的输出，因此我们将它们展平
        # 方案：将 patch 和 cls 按顺序拼接为一个 tensor 列表，并分别命名
        outputs = []
        for i, (p, c) in enumerate(zip(patch_tokens, cls_tokens)):
            outputs.append(p)   # patch token for layer i
            outputs.append(c)   # cls token for layer i
        return outputs

# 要导出的层（与构建记忆库时一致）
layer_indices = [3, 6, 9, 11]
wrapper = DINOv3Wrapper(model, layer_indices)

# 准备 dummy 输入（需与训练时预处理一致：448x448，归一化方式不变）
dummy_input = torch.randn(1, 3, 448, 448)

# 定义输出名称
output_names = []
for idx in layer_indices:
    output_names.append(f'patch_{idx}')
    output_names.append(f'cls_{idx}')

# 动态轴：batch 维度可变
dynamic_axes = {'input': {0: 'batch_size'}}
for name in output_names:
    dynamic_axes[name] = {0: 'batch_size'}

# 导出 ONNX
torch.onnx.export(
    wrapper,
    dummy_input,
    "dinov3_multilayer.onnx",
    input_names=['input'],
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=14,          # 推荐 14 以上
    do_constant_folding=True,
    export_params=True
)

print("ONNX 模型导出成功！")