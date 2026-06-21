"""
ai_agent_deepfool.py
─────────────────────────────────────────────────────────────────────────────
Standard Grad-CAM masked DeepFool Adversarial Attack (No AI Agent)

- Fixed hyperparameters
- Max 1000 iterations
- No multi-round AI agent involvement
"""

import os
import csv
import sys
import logging
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
from skimage.metrics import structural_similarity as ssim_fn

# ─────────────────────────────────────────────
#  Global Configuration
# ─────────────────────────────────────────────
MODEL_PATH   = "../../data/cat_dog_classifier_resnet150.h5"
INPUT_DIR    = "../../data/Dog_image"
OUTPUT_DIR   = "../../results/1000_gradcam_technic_deepfool/Adversarial_Dog_Standard"
LOG_FILE     = "../../results/1000_gradcam_technic_deepfool/standard_attack_log.csv"

# Fixed Hyperparameters
EPSILON            = 0.05
OVERSHOOT          = 1.02
GRAD_CAM_THRESHOLD = 0.40
MAX_ITER           = 1000
SSIM_FLOOR         = 0.90     # Optional: Used to determine if "success" is high quality

IMG_SIZE     = 224

TARGET_CLASSES = ["Cat", "Dog"]   # index 0 = Cat, 1 = Dog

# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("standard_pipeline.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════
#  SECTION 1 — Grad-CAM Utilities
# ═════════════════════════════════════════════

def find_last_conv_layer(model):
    """Recursively find the last Conv2D in a (possibly nested) model."""
    def _search(m):
        for layer in reversed(m.layers):
            if hasattr(layer, "layers"):
                result = _search(layer)
                if result is not None:
                    return result
            if isinstance(layer, (tf.keras.layers.Conv2D,
                                  tf.keras.layers.DepthwiseConv2D)):
                return layer
        return None
    return _search(model)


def make_gradcam_mask(img_array, model, class_idx, last_conv_layer, threshold=0.4):
    """
    Returns a binary spatial mask (1, H, W, 1) highlighting pixels the model
    relies on most for its prediction, via Grad-CAM.
    """
    try:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            tape.watch(conv_out)
            loss = preds[0] if class_idx == 1 else (1.0 - preds[0])
        grads = tape.gradient(loss, conv_out)

    except Exception:
        backbone   = model.layers[0]
        head_layers = model.layers[1:]
        grad_model = tf.keras.Model(
            inputs=backbone.inputs,
            outputs=[last_conv_layer.output, backbone.output]
        )
        with tf.GradientTape() as tape:
            conv_out, bb_out = grad_model(img_array)
            tape.watch(conv_out)
            x = bb_out
            for lyr in head_layers:
                x = lyr(x)
            loss = x[0] if class_idx == 1 else (1.0 - x[0])
        grads = tape.gradient(loss, conv_out)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.nn.relu(heatmap)
    max_val  = tf.math.reduce_max(heatmap)
    heatmap  = heatmap / (max_val + 1e-10)
    heatmap_np = heatmap.numpy().astype(np.float32)

    heatmap_resized = cv2.resize(heatmap_np, (IMG_SIZE, IMG_SIZE))
    mask = (heatmap_resized > threshold).astype(np.float32)
    mask = mask[np.newaxis, :, :, np.newaxis]   # (1, H, W, 1)

    return mask, heatmap_resized


# ═════════════════════════════════════════════
#  SECTION 2 — DeepFool (single round)
# ═════════════════════════════════════════════

def deepfool_round(model, img_array, mask, epsilon, overshoot, max_iter):
    """
    Grad-CAM-masked DeepFool execution.
    """
    orig_img  = img_array.copy().astype(np.float32)
    orig_np   = orig_img[0]                          # (H, W, 3)
    x         = tf.Variable(orig_img, dtype=tf.float32)
    r_tot     = np.zeros_like(orig_img, dtype=np.float32)

    prob0      = float(model(x).numpy()[0, 0])
    orig_label = 1 if prob0 > 0.5 else 0

    final_ssim = 1.0
    iters_used = max_iter

    for i in range(max_iter):
        with tf.GradientTape() as tape:
            tape.watch(x)
            prob = model(x)[0, 0]
            f_x  = prob - 0.5

        grad = tape.gradient(f_x, x)
        if grad is None:
            log.warning("Gradient is None — stopping early.")
            break

        grad_np = grad.numpy().astype(np.float32)
        f_val   = float(f_x.numpy())

        # Standard DeepFool step
        norm_sq = float(np.sum(grad_np ** 2)) + 1e-8
        r_i     = -(f_val / norm_sq) * grad_np

        # Apply spatial mask
        r_i   = r_i * mask
        r_tot += r_i

        # L∞ clamp
        r_clipped = np.clip(r_tot, -epsilon, epsilon).astype(np.float32)

        # Overshoot and project back to [0, 1]
        adv = np.clip(orig_img + overshoot * r_clipped, 0.0, 1.0)
        x.assign(adv)

        new_prob  = float(model(x).numpy()[0, 0])
        new_label = 1 if new_prob > 0.5 else 0

        adv_np = x.numpy()[0]
        ssim_v, _ = ssim_fn(orig_np, adv_np,
                            full=True, channel_axis=2, data_range=1.0)
        final_ssim = float(ssim_v)

        if new_label != orig_label:
            iters_used = i + 1
            return x.numpy(), iters_used, new_label, new_prob, final_ssim

    final_prob  = float(model(x).numpy()[0, 0])
    final_label = 1 if final_prob > 0.5 else 0
    return x.numpy(), iters_used, final_label, final_prob, final_ssim


# ═════════════════════════════════════════════
#  SECTION 3 — Main Pipeline
# ═════════════════════════════════════════════

def main():
    # ── Sanity checks ──────────────────────────────
    for path, label in [(MODEL_PATH, "Model"), (INPUT_DIR, "Input directory")]:
        if not os.path.exists(path):
            log.error(f"{label} not found: {path}")
            return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load model ─────────────────────────────────
    log.info(f"Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    log.info("Model loaded.")

    last_conv = find_last_conv_layer(model)
    if last_conv is None:
        log.warning("No Conv2D found — using full-image mask.")
    else:
        log.info(f"Last conv layer: {last_conv.name}")

    # ── Image list ─────────────────────────────────
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = sorted(
        f for f in os.listdir(INPUT_DIR) if f.lower().endswith(exts)
    )
    log.info(f"Found {len(image_files)} images.")

    # ── CSV setup ──────────────────────────
    csv_cols = [
        "image", "original_label", "original_confidence",
        "adversarial_label", "adversarial_confidence",
        "total_iterations", "fooled", "final_SSIM",
        "epsilon", "overshoot", "grad_cam_threshold"
    ]

    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=csv_cols)
        writer.writeheader()

        for img_idx, img_name in enumerate(image_files):
            img_path = os.path.join(INPUT_DIR, img_name)
            log.info(f"\n{'='*60}")
            log.info(f"[{img_idx+1}/{len(image_files)}]  {img_name}")

            try:
                # ── Load & preprocess ──────────────────────
                img    = keras_image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                img_np = keras_image.img_to_array(img) / 255.0
                img_batch = np.expand_dims(img_np, 0).astype(np.float32)

                prob0      = float(model.predict(img_batch, verbose=0)[0, 0])
                orig_idx   = 1 if prob0 > 0.5 else 0
                orig_label = TARGET_CLASSES[orig_idx]
                orig_conf  = prob0 if orig_idx == 1 else (1.0 - prob0)
                log.info(f"  Original → {orig_label} (sigmoid={prob0:.4f}, conf={orig_conf:.4f})")

                # ── Build Grad-CAM mask  ───────────────────
                if last_conv is not None:
                    mask, _ = make_gradcam_mask(
                        img_batch, model, orig_idx, last_conv,
                        threshold=GRAD_CAM_THRESHOLD
                    )
                else:
                    mask = np.ones((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

                mask_pct = float(mask.mean()) * 100
                log.info(f"  Grad-CAM mask coverage: {mask_pct:.1f}%")

                log.info(
                    f"  Starting DeepFool → ε={EPSILON:.3f}, "
                    f"overshoot={OVERSHOOT:.3f}, "
                    f"gcam_thresh={GRAD_CAM_THRESHOLD:.2f}, "
                    f"max_iter={MAX_ITER}"
                )

                # ── Run DeepFool  ──────────────────────────
                adv_batch, iters, adv_idx, adv_prob, ssim_val = deepfool_round(
                    model, img_batch, mask,
                    epsilon   = EPSILON,
                    overshoot = OVERSHOOT,
                    max_iter  = MAX_ITER,
                )

                fooled       = (adv_idx != orig_idx)
                adv_label    = TARGET_CLASSES[adv_idx]
                
                log.info(
                    f"  Result → {adv_label} (sigmoid={adv_prob:.4f}) | "
                    f"Fooled={fooled} | SSIM={ssim_val:.4f} | Iters={iters}"
                )

                # ── Save final adversarial image ───────────
                adv_uint8 = (np.clip(adv_batch[0], 0, 1) * 255).astype(np.uint8)
                base_name = os.path.splitext(img_name)[0]
                out_path  = os.path.join(OUTPUT_DIR, f"{base_name}_standard_adv.png")
                cv2.imwrite(out_path, cv2.cvtColor(adv_uint8, cv2.COLOR_RGB2BGR))

                # ── CSV row ────────────────────────────────
                success = fooled and (ssim_val >= SSIM_FLOOR)

                writer.writerow({
                    "image":                img_name,
                    "original_label":       orig_label,
                    "original_confidence":  f"{orig_conf:.4f}",
                    "adversarial_label":    adv_label,
                    "adversarial_confidence": f"{adv_prob:.4f}",
                    "total_iterations":     iters,
                    "fooled":               str(success),
                    "final_SSIM":           f"{ssim_val:.4f}",
                    "epsilon":              f"{EPSILON}",
                    "overshoot":            f"{OVERSHOOT}",
                    "grad_cam_threshold":   f"{GRAD_CAM_THRESHOLD}",
                })
                f_csv.flush()

            except Exception as exc:
                log.exception(f"  ERROR on {img_name}: {exc}")
                writer.writerow({
                    "image": img_name,
                    "original_label": "ERROR",
                    "original_confidence": "N/A",
                    "adversarial_label": "ERROR",
                    "adversarial_confidence": "N/A",
                    "total_iterations": 0,
                    "fooled": "False",
                    "final_SSIM": "N/A",
                    "epsilon": "N/A",
                    "overshoot": "N/A",
                    "grad_cam_threshold": "N/A",
                })
                f_csv.flush()

    log.info(f"\n✓ Done. Adversarial images → '{OUTPUT_DIR}/'")
    log.info(f"   Attack CSV   → '{LOG_FILE}'")


if __name__ == "__main__":
    main()
