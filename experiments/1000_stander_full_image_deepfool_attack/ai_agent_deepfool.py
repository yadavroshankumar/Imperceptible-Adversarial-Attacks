"""
ai_agent_deepfool.py
─────────────────────────────────────────────────────────────────────────────
Standard DeepFool Adversarial Attack (Full Image)

This script performs a standard DeepFool adversarial attack with fixed 
hyperparameters, applying noise across the entire image (no Grad-CAM).
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
OUTPUT_DIR   = "../../results/1000_stander_full_image_deepfool_attack/Adversarial_Dog_Standard"
LOG_FILE     = "../../results/1000_stander_full_image_deepfool_attack/standard_attack_log.csv"

EPSILON            = 0.0314
OVERSHOOT          = 1.02
MAX_ITER           = 1000
SSIM_FLOOR         = 0.90     # Optional: Used to determine if "success" is high quality
IMG_SIZE     = 224

TARGET_CLASSES = ["Cat", "Dog"]   # index 0 = Cat, 1 = Dog

# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────
if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

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
#  SECTION 1 — DeepFool (single round)
# ═════════════════════════════════════════════

def deepfool_round(model, img_array, epsilon, overshoot, max_iter):
    """
    One round of full-image DeepFool.

    Uses the standard DeepFool binary update:
        r_i = -f(x) / ‖∇f(x)‖²  ·  ∇f(x)

    where f(x) = sigmoid(x) - 0.5

    Perturbation is:
        - L∞-clipped to epsilon per pixel
        - Scaled by overshoot before applying to the image

    Returns
    ───────
    adv_img    : (1, H, W, 3) float32 adversarial image
    iters_used : int  (iterations until flip or max_iter)
    new_label  : int  (0 or 1)
    new_prob   : float (raw sigmoid output of adversarial image)
    ssim_val   : float (SSIM between original and adversarial)
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

        # Safety: stop if SSIM is already very low (< 0.85) — no point continuing
        if final_ssim < 0.85:
            iters_used = i + 1
            break

    final_prob  = float(model(x).numpy()[0, 0])
    final_label = 1 if final_prob > 0.5 else 0
    return x.numpy(), iters_used, final_label, final_prob, final_ssim


# ═════════════════════════════════════════════
#  SECTION 2 — Main Pipeline
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
        "iterations", "fooled", "final_SSIM"
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

                # ── Run Standard DeepFool ─────────────────────────────
                log.info(f"  Running standard DeepFool.")
                adv_batch, iters, adv_idx, adv_prob, ssim_val = deepfool_round(
                    model, img_batch,
                    epsilon   = EPSILON,
                    overshoot = OVERSHOOT,
                    max_iter  = MAX_ITER,
                )

                fooled       = (adv_idx != orig_idx)
                adv_label    = TARGET_CLASSES[adv_idx]
                adv_conf     = adv_prob if adv_idx == 1 else (1.0 - adv_prob)

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
                success = fooled and ssim_val >= SSIM_FLOOR
                if success:
                    log.info(f"  ✓ SUCCESS! Fooled={fooled}, SSIM={ssim_val:.4f} >= {SSIM_FLOOR}")
                elif fooled and ssim_val < SSIM_FLOOR:
                    log.info(f"  ~ Fooled but SSIM={ssim_val:.4f} < {SSIM_FLOOR}")

                writer.writerow({
                    "image":                img_name,
                    "original_label":       orig_label,
                    "original_confidence":  f"{orig_conf:.4f}",
                    "adversarial_label":    adv_label,
                    "adversarial_confidence": f"{adv_conf:.4f}",
                    "iterations":           iters,
                    "fooled":               str(success),
                    "final_SSIM":           f"{ssim_val:.4f}"
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
                    "iterations": 0,
                    "fooled": "False",
                    "final_SSIM": "N/A",
                })
                f_csv.flush()

    log.info(f"\n✓ Done. Adversarial images → '{OUTPUT_DIR}/'")
    log.info(f"   Attack CSV   → '{LOG_FILE}'")


if __name__ == "__main__":
    main()
