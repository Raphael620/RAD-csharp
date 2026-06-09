"""
RAD (Retrieval-based Anomaly Detection) using OpenCV 5 DNN + DINOv2.
Loads dinov2_small.onnx via OpenCV 5's new DNN engine, builds a memory bank
from normal samples, then scores test images via multi-layer patch-KNN.
"""
import os, sys, time, struct, argparse
import numpy as np

# ──────────────────────────────────────────────────────────────────
#  OpenCV 5 环境设置 (exe 安装版, 非 pip) — MUST run before `import cv2`
# ──────────────────────────────────────────────────────────────────
_build_dir = os.path.expandvars(r"%USERPROFILE%\Downloads\opencv5\opencv\build")
_dll_dir = os.path.join(_build_dir, "x64", "vc16", "bin")
_pyd_dir = os.path.join(_build_dir, "python", "cv2", "python-3.13")
if os.path.isdir(_dll_dir):
    os.add_dll_directory(_dll_dir)
if os.path.isdir(_pyd_dir) and _pyd_dir not in sys.path:
    sys.path.insert(0, _pyd_dir)

import cv2

# ImageNet stats
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# ──────────────────────────────────────────────────────────────────
#  Image preprocessing (matches RAD-main/dataset.py & C# ImagePreprocessor)
# ──────────────────────────────────────────────────────────────────
def preprocess(image_path: str, crop_size: int = 448, resize_size: int = 512) -> np.ndarray:
    """Load image, resize→centercrop→normalize → NCHW float32."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
    offset = (resize_size - crop_size) // 2
    img = img[offset:offset + crop_size, offset:offset + crop_size]
    # To float [0,1] then normalize
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    # HWC -> CHW -> add batch dim
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0).astype(np.float32)

# ──────────────────────────────────────────────────────────────────
#  Binary bank I/O (compatible with bank_pth2bin.py + C# BankSerializer)
# ──────────────────────────────────────────────────────────────────
def read_2d_bin(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        ndim = struct.unpack("i", f.read(4))[0]
        assert ndim == 2
        rows = struct.unpack("i", f.read(4))[0]
        cols = struct.unpack("i", f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(rows, cols)
    return data

def read_3d_bin(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        ndim = struct.unpack("i", f.read(4))[0]
        assert ndim == 3
        d1 = struct.unpack("i", f.read(4))[0]
        d2 = struct.unpack("i", f.read(4))[0]
        d3 = struct.unpack("i", f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(d1, d2, d3)
    return data

def read_layers_bin(path: str) -> list[int]:
    with open(path, "rb") as f:
        count = struct.unpack("i", f.read(4))[0]
        layers = [struct.unpack("i", f.read(4))[0] for _ in range(count)]
    return layers

def write_2d_bin(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("i", 2))
        f.write(struct.pack("i", arr.shape[0]))
        f.write(struct.pack("i", arr.shape[1]))
        f.write(arr.astype(np.float32).tobytes())

def write_3d_bin(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("i", 3))
        f.write(struct.pack("i", arr.shape[0]))
        f.write(struct.pack("i", arr.shape[1]))
        f.write(struct.pack("i", arr.shape[2]))
        f.write(arr.astype(np.float32).tobytes())

def write_layers_bin(path: str, layers: list[int]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("i", len(layers)))
        for l in layers:
            f.write(struct.pack("i", l))

# ──────────────────────────────────────────────────────────────────
#  OpenCV 5 DINOv2 wrapper
# ──────────────────────────────────────────────────────────────────
class DinoV2OpenCV:
    """Load dinov2_small.onnx with OpenCV 5 DNN and run inference."""

    def __init__(self, onnx_path: str, engine: int = 3):
        # engine: 3=AUTO (new parser, fallback classic), 2=NEW only
        self.net = cv2.dnn.readNetFromONNX(onnx_path, engine=engine)
        print(f"[DinoV2] Loaded {onnx_path}")

    def infer(self, blob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (cls_token, patch_tokens).
        - cls_token: [C]   (first token)
        - patch_tokens: [L, C]  (remaining tokens)
        """
        self.net.setInput(blob)
        out = self.net.forward()  # [1, 1+L, C]
        if isinstance(out, (tuple, list)):
            out = out[0]
        data = np.array(out)
        cls_token = data[0, 0, :]           # [C]
        patch_tokens = data[0, 1:, :]       # [L, C]
        return cls_token, patch_tokens

# ──────────────────────────────────────────────────────────────────
#  RAD: Bank building
# ──────────────────────────────────────────────────────────────────
def build_bank(
    encoder: DinoV2OpenCV,
    image_dir: str,
    bank_output_dir: str,
    onnx_path: str = "model/dinov2_small.onnx",
    engine: int = 3,
):
    """Process all images in image_dir and save bank to bank_output_dir."""
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    image_files = sorted(
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in exts
    )
    if not image_files:
        raise ValueError(f"No images in {image_dir}")

    cls_feats = []
    patch_feats = []

    for i, fname in enumerate(image_files):
        path = os.path.join(image_dir, fname)
        print(f"  [{i+1}/{len(image_files)}] {fname}")
        try:
            blob = preprocess(path)
            cls_token, patch_tokens = encoder.infer(blob)
            cls_feats.append(cls_token)
            patch_feats.append(patch_tokens)
        except Exception as e:
            print(f"    SKIP: {e}")

    N = len(cls_feats)
    if N == 0:
        raise RuntimeError("No images processed successfully")

    C = cls_feats[0].shape[0]
    L = patch_feats[0].shape[0]

    cls_bank = np.stack(cls_feats, axis=0)      # [N, C]
    patch_bank = np.stack(patch_feats, axis=0)  # [N, L, C]

    layers = [0]  # single-layer model
    write_layers_bin(os.path.join(bank_output_dir, "layers.bin"), layers)
    write_2d_bin(os.path.join(bank_output_dir, "cls_layer0.bin"), cls_bank)
    write_3d_bin(os.path.join(bank_output_dir, "patch_layer0.bin"), patch_bank)

    print(f"[Bank] Built: {N} images, L={L}, C={C} → {bank_output_dir}")
    return cls_bank, patch_bank

# ──────────────────────────────────────────────────────────────────
#  RAD: load pre-built bank
# ──────────────────────────────────────────────────────────────────
def load_bank(bank_dir: str):
    layers = read_layers_bin(os.path.join(bank_dir, "layers.bin"))
    cls_banks = []
    patch_banks = []
    patch_banks_norm = []
    for l in layers:
        cb = read_2d_bin(os.path.join(bank_dir, f"cls_layer{l}.bin"))
        pb = read_3d_bin(os.path.join(bank_dir, f"patch_layer{l}.bin"))
        # Pre-normalize each patch vector to unit length
        pb_norm_base = pb / (np.linalg.norm(pb, axis=2, keepdims=True) + 1e-8)
        cls_banks.append(cb)
        patch_banks.append(pb)
        patch_banks_norm.append(pb_norm_base)
    return layers, cls_banks, patch_banks, patch_banks_norm

# ──────────────────────────────────────────────────────────────────
#  RAD: Inference (image-level retrieval + patch-KNN)
# ──────────────────────────────────────────────────────────────────
def detect(
    encoder: DinoV2OpenCV,
    cls_banks: list[np.ndarray],
    patch_banks_norm: list[np.ndarray],
    image_path: str,
    k_image: int = 5,
    output_size: int = 256,
    apply_smoothing: bool = True,
    layer_weights: list[float] | None = None,
):
    """
    Run RAD anomaly detection on a single image.

    Returns (anomaly_map [H,W], image_score float).
    """
    num_layers = len(cls_banks)
    embed_dim = cls_banks[0].shape[1]
    num_patches = patch_banks_norm[0].shape[1]

    # Weights
    if layer_weights is None:
        layer_weights = [1.0 / num_layers] * num_layers

    # 1) Forward pass
    blob = preprocess(image_path)
    query_cls, query_patch = encoder.infer(blob)  # [C], [L, C]

    # 2) Image-level retrieval (use last layer CLS)
    cls_bank_global = cls_banks[-1]                # [N, C]
    query_cls_norm = query_cls / (np.linalg.norm(query_cls) + 1e-8)

    # Cosine similarity: dot(query_norm, bank[i]) / ||bank[i]||
    bank_norms = np.linalg.norm(cls_bank_global, axis=1) + 1e-8
    sims = np.dot(cls_bank_global, query_cls_norm) / bank_norms  # [N]
    topk = np.argsort(sims)[-min(k_image, len(sims)):][::-1]     # [k_image] descending

    # 3) Multi-layer patch-KNN
    fused_scores = np.zeros(num_patches, dtype=np.float32)

    for li in range(num_layers):
        q_patch = query_patch if li == 0 else query_patch  # [L, C]; single-layer → same
        q_patch_norm = q_patch / (np.linalg.norm(q_patch, axis=1, keepdims=True) + 1e-8)

        bank_patch_norm = patch_banks_norm[li][topk]  # [k, L, C]
        bank_flat = bank_patch_norm.reshape(-1, embed_dim)  # [k*L, C]

        # For each query patch, find max cosine similarity via matrix multiply
        # [L, C] @ [k*L, C]^T = [L, k*L], then max over axis=1
        sims = np.dot(q_patch_norm, bank_flat.T)                   # [L, k*L]
        max_sims = sims.max(axis=1)                                 # [L]
        layer_scores = 1.0 - max_sims

        fused_scores += layer_weights[li] * layer_scores

    # 4) Reshape to 2D map + bilinear upsample
    patch_h = int(np.sqrt(num_patches))
    patch_w = patch_h
    patch_map = fused_scores.reshape(patch_h, patch_w)

    anomaly_map = cv2.resize(
        patch_map, (output_size, output_size), interpolation=cv2.INTER_LINEAR
    )

    # 5) Gaussian smoothing
    if apply_smoothing:
        anomaly_map = cv2.GaussianBlur(anomaly_map, (5, 5), sigmaX=1.0)

    score = float(np.max(anomaly_map))
    return anomaly_map, score


# ──────────────────────────────────────────────────────────────────
#  Visualization
# ──────────────────────────────────────────────────────────────────
def visualize(original_path: str, anomaly_map: np.ndarray, score: float, save_path: str | None = None):
    """Overlay anomaly heatmap on original image."""
    img = cv2.imread(original_path)
    if img is None:
        print(f"  Cannot read {original_path} for visualization")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize anomaly_map to match original
    h, w = img.shape[:2]
    am_resized = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)
    am_norm = (am_resized - am_resized.min()) / (am_resized.max() - am_resized.min() + 1e-8)
    am_uint8 = (am_norm * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(am_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.5, heatmap, 0.5, 0)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"Overlay (score={score:.4f})")
    axes[1].axis("off")

    im = axes[2].imshow(am_norm, cmap="jet")
    axes[2].set_title("Anomaly Map")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RAD anomaly detection with OpenCV 5 + DINOv2")
    parser.add_argument("--onnx", default="model/dinov2_small.onnx", help="Path to DINOv2 ONNX model")
    parser.add_argument("--bank-dir", default="model/bank_dinov2", help="Bank output/input directory")
    parser.add_argument("--normal-dir", default="sample/OK", help="Directory of normal images for bank")
    parser.add_argument("--test-image", default=None, help="Single test image path")
    parser.add_argument("--test-dir", default=None, help="Directory of test images")
    parser.add_argument("--k-image", type=int, default=3, help="Top-K NN images for local patch bank")
    parser.add_argument("--build-only", action="store_true", help="Only build bank, no detection")
    parser.add_argument("--engine", type=int, default=3,
                        help="OpenCV DNN engine: 3=AUTO, 2=NEW, 1=CLASSIC")
    parser.add_argument("--output-dir", default="output", help="Directory for visualization results")
    args = parser.parse_args()

    print("=" * 60)
    print("RAD Anomaly Detection — OpenCV 5 + DINOv2")
    print(f"  ONNX:      {args.onnx}")
    print(f"  Bank dir:  {args.bank_dir}")
    print(f"  K-image:   {args.k_image}")
    print(f"  Engine:    {args.engine}")
    print("=" * 60)

    if not os.path.exists(args.onnx):
        print(f"ERROR: ONNX model not found: {args.onnx}")
        sys.exit(1)

    # ── Load encoder ──
    print("\n[1] Loading encoder...")
    encoder = DinoV2OpenCV(args.onnx, engine=args.engine)

    # Test forward pass
    print("[2] Smoke test forward pass...")
    try:
        test_blob = preprocess(os.path.join(args.normal_dir, os.listdir(args.normal_dir)[0]))
        cls_tok, patch_tok = encoder.infer(test_blob)
        print(f"    CLS shape: {cls_tok.shape}, Patch shape: {patch_tok.shape}")
    except Exception as e:
        print(f"    FORWARD FAILED: {e}")
        sys.exit(1)

    # ── Build bank ──
    print(f"\n[3] Building bank from: {args.normal_dir}")
    cls_bank, patch_bank = build_bank(
        encoder, args.normal_dir, args.bank_dir,
        onnx_path=args.onnx, engine=args.engine,
    )

    # Load bank back (to get pre-normalized patch banks)
    layers, cls_banks, patch_banks, patch_banks_norm = load_bank(args.bank_dir)
    print(f"    Layers: {layers}, CLS bank: {cls_banks[0].shape}, Patch bank: {patch_banks[0].shape}")

    if args.build_only:
        print("\n[Done] Bank built. Exiting (--build-only).")
        return

    # ── Detect ──
    os.makedirs(args.output_dir, exist_ok=True)
    k_image = min(args.k_image, cls_banks[0].shape[0])

    if args.test_image:
        test_paths = [args.test_image]
    elif args.test_dir:
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        test_paths = sorted(
            os.path.join(args.test_dir, f) for f in os.listdir(args.test_dir)
            if os.path.splitext(f)[1].lower() in exts
        )
    else:
        # Default: use normal images as a quick sanity test (should get low scores)
        print("\n[WARN] No test image specified. Using normal samples for sanity check.")
        test_paths = [os.path.join(args.normal_dir, f) for f in os.listdir(args.normal_dir)[:3]
                      if os.path.splitext(f)[1].lower() in {".png", ".jpg", ".jpeg", ".bmp"}]

    print(f"\n[4] Running detection on {len(test_paths)} image(s)...")
    for path in test_paths:
        name = os.path.basename(path)
        print(f"\n  --- {name} ---")
        t0 = time.perf_counter()
        amap, score = detect(
            encoder, cls_banks, patch_banks_norm,
            path, k_image=k_image, output_size=256, apply_smoothing=True,
        )
        dt = time.perf_counter() - t0
        print(f"    Score: {score:.4f}  Time: {dt*1000:.1f}ms")

        save_path = os.path.join(args.output_dir, f"rad_{name}")
        visualize(path, amap, score, save_path=save_path)

    print("\n[Done]")


if __name__ == "__main__":
    main()
