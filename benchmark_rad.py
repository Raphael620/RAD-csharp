"""
Speed benchmark: OpenCV 5 ENGINE_NEW with DINOv2, timing preprocess / DINO forward / post-KNN.
Usage: python benchmark_rad.py
"""
import os, sys, time, struct, argparse
import numpy as np

# ── OpenCV 5 setup (must precede `import cv2`) ──
_build_dir = os.path.expandvars(r"%USERPROFILE%\Downloads\opencv5\opencv\build")
_dll_dir = os.path.join(_build_dir, "x64", "vc16", "bin")
_pyd_dir = os.path.join(_build_dir, "python", "cv2", "python-3.13")
if os.path.isdir(_dll_dir):
    os.add_dll_directory(_dll_dir)
if os.path.isdir(_pyd_dir) and _pyd_dir not in sys.path:
    sys.path.insert(0, _pyd_dir)

import cv2

# ── Constants ──
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# ── Preprocess ──
def preprocess(image_path: str, crop_size: int = 448, resize_size: int = 512) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
    offset = (resize_size - crop_size) // 2
    img = img[offset:offset + crop_size, offset:offset + crop_size]
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0).astype(np.float32)

# ── Binary bank I/O ──
import struct as _struct
def read_2d_bin(path): 
    with open(path,"rb") as f:
        ndim=_struct.unpack("i",f.read(4))[0]; assert ndim==2
        r=_struct.unpack("i",f.read(4))[0]; c=_struct.unpack("i",f.read(4))[0]
        return np.frombuffer(f.read(),dtype=np.float32).reshape(r,c)
def read_3d_bin(path):
    with open(path,"rb") as f:
        ndim=_struct.unpack("i",f.read(4))[0]; assert ndim==3
        d1=_struct.unpack("i",f.read(4))[0]; d2=_struct.unpack("i",f.read(4))[0]; d3=_struct.unpack("i",f.read(4))[0]
        return np.frombuffer(f.read(),dtype=np.float32).reshape(d1,d2,d3)
def read_layers_bin(path):
    with open(path,"rb") as f:
        count=_struct.unpack("i",f.read(4))[0]
        return [_struct.unpack("i",f.read(4))[0] for _ in range(count)]
def write_2d_bin(path,arr):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path,"wb") as f:
        f.write(_struct.pack("i",2)); f.write(_struct.pack("i",arr.shape[0])); f.write(_struct.pack("i",arr.shape[1]))
        f.write(arr.astype(np.float32).tobytes())
def write_3d_bin(path,arr):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path,"wb") as f:
        f.write(_struct.pack("i",3)); f.write(_struct.pack("i",arr.shape[0])); f.write(_struct.pack("i",arr.shape[1])); f.write(_struct.pack("i",arr.shape[2]))
        f.write(arr.astype(np.float32).tobytes())
def write_layers_bin(path,layers):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path,"wb") as f:
        f.write(_struct.pack("i",len(layers)))
        for l in layers: f.write(_struct.pack("i",l))

# ── DINOv2 via OpenCV 5 ──
class DinoV2OCV:
    def __init__(self, onnx_path, engine=2):
        self.net = cv2.dnn.readNetFromONNX(onnx_path, engine=engine)
    def infer(self, blob):
        self.net.setInput(blob)
        out = self.net.forward()
        if isinstance(out, (tuple,list)): out = out[0]
        data = np.array(out)
        return data[0,0,:], data[0,1:,:]  # cls, patches

# ── Build bank ──
def build_bank(encoder, image_dir, bank_dir):
    exts = {".png",".jpg",".jpeg",".bmp"}
    files = sorted(f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in exts)
    cls_list, patch_list = [], []
    for f in files:
        blob = preprocess(os.path.join(image_dir, f))
        c, p = encoder.infer(blob)
        cls_list.append(c); patch_list.append(p)
    cls_bank = np.stack(cls_list,0); patch_bank = np.stack(patch_list,0)
    write_layers_bin(os.path.join(bank_dir,"layers.bin"), [0])
    write_2d_bin(os.path.join(bank_dir,"cls_layer0.bin"), cls_bank)
    write_3d_bin(os.path.join(bank_dir,"patch_layer0.bin"), patch_bank)
    return cls_bank, patch_bank

def load_bank(bank_dir):
    layers = read_layers_bin(os.path.join(bank_dir,"layers.bin"))
    cb = read_2d_bin(os.path.join(bank_dir,f"cls_layer{layers[0]}.bin"))
    pb = read_3d_bin(os.path.join(bank_dir,f"patch_layer{layers[0]}.bin"))
    pb_norm = pb / (np.linalg.norm(pb, axis=2, keepdims=True) + 1e-8)
    return cb, pb_norm, cb.shape[1], pb.shape[1]

# ── Detection (timed) ──
def detect(encoder, cls_bank, patch_bank_norm, image_path, k_image, embed_dim, num_patches):
    """Returns (anomaly_map, score, pre_ms, bone_ms, post_ms)."""
    # --- Preprocess ---
    t0 = time.perf_counter()
    blob = preprocess(image_path)
    pre_ms = (time.perf_counter() - t0) * 1000

    # --- DINO forward ---
    t0 = time.perf_counter()
    query_cls, query_patch = encoder.infer(blob)
    bone_ms = (time.perf_counter() - t0) * 1000

    # --- Post: KNN ---
    t0 = time.perf_counter()

    # Image-level retrieval
    query_cls_norm = query_cls / (np.linalg.norm(query_cls) + 1e-8)
    bank_norms = np.linalg.norm(cls_bank, axis=1) + 1e-8
    sims = np.dot(cls_bank, query_cls_norm) / bank_norms
    k = min(k_image, len(sims))
    topk = np.argsort(sims)[-k:][::-1]

    # Patch-KNN (single layer) — vectorized: [L,C] @ [C,k*L] → [L,k*L]
    q_norm = query_patch / (np.linalg.norm(query_patch, axis=1, keepdims=True) + 1e-8)
    bank_flat = patch_bank_norm[topk].reshape(-1, embed_dim)     # [k*L, C]
    sims = np.dot(q_norm, bank_flat.T)                           # [L, k*L]
    max_sims = sims.max(axis=1)                                  # [L]
    layer_scores = 1.0 - max_sims

    # Reshape + upsample + smooth
    patch_h = int(np.sqrt(num_patches))
    patch_map = layer_scores.reshape(patch_h, patch_h)
    anomaly_map = cv2.resize(patch_map, (256, 256), interpolation=cv2.INTER_LINEAR)
    anomaly_map = cv2.GaussianBlur(anomaly_map, (5, 5), sigmaX=1.0)

    post_ms = (time.perf_counter() - t0) * 1000
    score = float(np.max(anomaly_map))
    return anomaly_map, score, pre_ms, bone_ms, post_ms

# ── Main ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="model/dinov2_small.onnx")
    parser.add_argument("--bank-dir", default="model/bank_dinov2")
    parser.add_argument("--normal-dir", default="sample/OK")
    parser.add_argument("--ng-dir", default="sample/NG")
    parser.add_argument("--k-image", type=int, default=10)
    parser.add_argument("--engine", type=int, default=2, help="2=ENGINE_NEW, 3=AUTO")
    args = parser.parse_args()

    print("=" * 60)
    print("RAD Speed Benchmark — OpenCV 5 ENGINE_NEW + DINOv2")
    print(f"  ONNX: {args.onnx} | Engine: {args.engine} | K: {args.k_image}")
    print("=" * 60)

    # Load encoder
    encoder = DinoV2OCV(args.onnx, engine=args.engine)

    # Build bank
    print(f"\n[1] Build bank from {args.normal_dir}")
    cls_bank, patch_bank = build_bank(encoder, args.normal_dir, args.bank_dir)
    N = cls_bank.shape[0]
    print(f"    Bank: {N} images, patches={patch_bank.shape[1]}, dim={patch_bank.shape[2]}")

    # Load + pre-normalize
    cls_bank, patch_bank_norm, embed_dim, num_patches = load_bank(args.bank_dir)

    # Find NG images
    exts = {".png",".jpg",".jpeg",".bmp"}
    ng_files = sorted(f for f in os.listdir(args.ng_dir) if os.path.splitext(f)[1].lower() in exts)
    if len(ng_files) < 3:
        print(f"ERROR: need >=3 NG images, got {len(ng_files)}")
        return

    print(f"\n[2] Warmup: 2 images, then measure the 3rd")
    print("-" * 60)

    # Warmup
    for i in range(2):
        path = os.path.join(args.ng_dir, ng_files[i])
        _, _, pre_ms, bone_ms, post_ms = detect(
            encoder, cls_bank, patch_bank_norm, path,
            args.k_image, embed_dim, num_patches)
        print(f"  Warmup [{i+1}] {ng_files[i]}: pre={pre_ms:.1f}ms  dino={bone_ms:.1f}ms  post={post_ms:.1f}ms  total={pre_ms+bone_ms+post_ms:.1f}ms")

    # Measure 3rd
    path = os.path.join(args.ng_dir, ng_files[2])
    amap, score, pre_ms, bone_ms, post_ms = detect(
        encoder, cls_bank, patch_bank_norm, path,
        args.k_image, embed_dim, num_patches)
    total = pre_ms + bone_ms + post_ms

    print("-" * 60)
    print(f"  [MEASURED] {ng_files[2]}:")
    print(f"    Preprocess:  {pre_ms:8.1f} ms")
    print(f"    DINO forward:{bone_ms:8.1f} ms")
    print(f"    Post (KNN):  {post_ms:8.1f} ms")
    print(f"    ─────────────────────")
    print(f"    TOTAL:       {total:8.1f} ms")
    print(f"    Score:       {score:8.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
