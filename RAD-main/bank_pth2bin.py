import torch
import numpy as np
import struct

# 加载记忆库
bank = torch.load('./bank/mvtec_dinov3_vits16_multilayer36911_448_bank.pth', map_location='cpu')
layers = bank['layers']            # [3,6,9,11]
cls_banks = bank['cls_banks']      # list of [N, C]
patch_banks = bank['patch_banks']  # list of [N, L, C]

def save_array_as_bin(arr, filename):
    """
    将 numpy 数组保存为 bin 文件，格式：
        [int32] 维度数量 (ndim)
        [int32] 每个维度的大小 (重复 ndim 次)
        [float32] 所有数据（按行主序，即 C-order）
    """
    with open(filename, 'wb') as f:
        # 写入维度数
        f.write(struct.pack('i', arr.ndim))
        # 写入每个维度的长度
        for dim in arr.shape:
            f.write(struct.pack('i', dim))
        # 写入数据（注意：numpy 默认 C-order，直接 tobytes() 即可）
        f.write(arr.astype(np.float32).tobytes())

# 保存每个层
for i, layer in enumerate(layers):
    cls_arr = cls_banks[i].numpy()      # [N, C]
    patch_arr = patch_banks[i].numpy()  # [N, L, C]
    save_array_as_bin(cls_arr, f'./bin/cls_layer{layer}.bin')
    save_array_as_bin(patch_arr, f'./bin/patch_layer{layer}.bin')

# 保存层索引（简单保存为 int 列表）
with open('layers.bin', 'wb') as f:
    f.write(struct.pack('i', len(layers)))
    for layer in layers:
        f.write(struct.pack('i', layer))

print("已转换为 .bin 文件")