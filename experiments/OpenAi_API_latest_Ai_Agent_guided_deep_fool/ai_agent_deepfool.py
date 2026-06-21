"""
ai_agent_deepfool.py
─────────────────────────────────────────────────────────────────────────────
LLM-Guided Adaptive DeepFool Adversarial Attack

The AI Agent (ChatGPT via OpenAI API) acts as a hyperparameter controller.
After each attack attempt it observes:
    - Model sigmoid output (confidence + distance to boundary 0.5)
    - SSIM value of the adversarial image vs original
    - Whether the model was fooled
    - How many iterations were consumed

It then decides the next set of hyperparameters to either:
    - Push harder  (SSIM > 0.95  → safely increase epsilon/overshoot)
    - Hold steady  (SSIM 0.90–0.95 → fine-tune)
    - Back off      (SSIM < 0.90  → reduce, attack was too aggressive)

SUCCESS condition : model fooled  AND  SSIM ≥ 0.90
FAILURE condition : max_rounds exhausted without success

Pipeline per image
──────────────────
  1. Load image, get original prediction & distance to boundary
  2. Agent Round 1 → Agent sees fresh image stats → picks initial params
  3. Run Grad-CAM masked DeepFool with those params
  4. Report (fooled, SSIM, iters, new_prob) back to Agent
  5. If success → log & move to next image
  6. If not success → Agent adjusts params → go to step 3
  7. Repeat up to MAX_ROUNDS

Hyperparameters controlled by the Agent
────────────────────────────────────────
  epsilon              : L∞ perturbation bound per pixel   [0.01 – 0.30]
  overshoot            : DeepFool overshoot multiplier     [1.00 – 1.20]
  grad_cam_threshold   : Fraction of heatmap to mask       [0.10 – 0.60]
  max_iter_per_round   : DeepFool iterations per round     [50  – 500 ]

Fixed constraints the Agent is always told
──────────────────────────────────────────
  SSIM_FLOOR = 0.90    (must never go below)
  Decision boundary at sigmoid = 0.50
"""

import os
import csv
import sys
import time
import json
import logging
import numpy as np
import tensorflow as tf
import cv2
from openai import OpenAI
from dotenv import load_dotenv
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
from skimage.metrics import structural_similarity as ssim_fn

# ─────────────────────────────────────────────
#  Global Configuration
# ─────────────────────────────────────────────
MODEL_PATH   = "../../data/cat_dog_classifier_resnet150.h5"
INPUT_DIR    = "../../data/Dog_image"
OUTPUT_DIR   = "../../results/OpenAi_API_latest_Ai_Agent_guided_deep_fool/Adversarial_Dog_AIAgent"
LOG_FILE     = "../../results/OpenAi_API_latest_Ai_Agent_guided_deep_fool/agent_attack_log.csv"
AGENT_LOG    = "agent_decisions.jsonl"     # every LLM decision recorded here

MAX_ROUNDS   = 6        # max LLM decision rounds per image
SSIM_FLOOR   = 0.90     # hard lower bound; agent must respect this
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
        logging.FileHandler("agent_pipeline.log", mode="w", encoding="utf-8"),
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
    One round of Grad-CAM-masked DeepFool.

    Uses the standard DeepFool binary update:
        r_i = -f(x) / ‖∇f(x)‖²  ·  ∇f(x)

    where f(x) = sigmoid(x) - 0.5

    Perturbation is:
        - Spatially restricted to the Grad-CAM mask
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

        # Safety: stop if SSIM is already very low (< 0.85) — no point continuing
        if final_ssim < 0.85:
            iters_used = i + 1
            break

    final_prob  = float(model(x).numpy()[0, 0])
    final_label = 1 if final_prob > 0.5 else 0
    return x.numpy(), iters_used, final_label, final_prob, final_ssim


# ═════════════════════════════════════════════
#  SECTION 3 — LLM Agent
# ═════════════════════════════════════════════

SYSTEM_PROMPT = """You are an adversarial attack optimization agent controlling a DeepFool 
attack on a binary image classifier (Cat vs Dog).

The classifier outputs a sigmoid probability:
  - Output >= 0.5  →  classified as Dog
  - Output  < 0.5  →  classified as Cat

Your task: decide hyperparameters that fool the model in as few total iterations as possible 
while keeping SSIM >= 0.90 (imperceptibility constraint).

Hyperparameters you control:
  epsilon            : L∞ perturbation per pixel  [0.01 – 0.30]
  overshoot          : DeepFool overshoot multiplier  [1.00 – 1.20]
  grad_cam_threshold : Grad-CAM mask threshold  [0.10 – 0.60]
  max_iter           : Max DeepFool iterations this round  [50 – 500]

Strategy rules you MUST follow:
  1. SSIM >= 0.90 is a hard constraint. Never sacrifice it.
  2. If SSIM is very high (> 0.97): the attack is too gentle. Increase epsilon and overshoot aggressively.
  3. If SSIM is 0.93–0.97: moderately increase epsilon and/or overshoot.
  4. If SSIM is 0.90–0.93: only small increases allowed; you are near the constraint boundary.
  5. If SSIM < 0.90: you overshot. Reduce epsilon significantly.
  6. The closer the model probability is to 0.5 (decision boundary), the easier it is to fool — use smaller epsilon in that case.
  7. If the model probability is far from 0.5 (e.g., 0.95), you need larger epsilon to push it across.
  8. Lower grad_cam_threshold means a larger spatial region is perturbed (more aggressive).

You must respond ONLY with a valid JSON object, no other text:
{
  "epsilon": <float>,
  "overshoot": <float>,
  "grad_cam_threshold": <float>,
  "max_iter": <int>,
  "reasoning": "<one concise sentence explaining your decision>"
}"""


def ask_agent(client, image_name, orig_label, orig_prob, round_num, history):
    """
    Calls the LLM agent with the current state and history.
    Returns a dict with epsilon, overshoot, grad_cam_threshold, max_iter, reasoning.

    history: list of dicts, each containing one round's result:
        { round, epsilon, overshoot, grad_cam_threshold, max_iter,
          fooled, ssim, new_prob, iters_used }
    """
    distance_to_boundary = abs(orig_prob - 0.5)

    user_msg = f"""Image: {image_name}
Original class: {orig_label}
Original sigmoid output: {orig_prob:.4f}
Distance to decision boundary (|prob - 0.5|): {distance_to_boundary:.4f}
Current round: {round_num} of {MAX_ROUNDS}
SSIM floor constraint: {SSIM_FLOOR}
"""

    if history:
        user_msg += "\nPrevious round results:\n"
        for h in history:
            user_msg += (
                f"  Round {h['round']}: "
                f"epsilon={h['epsilon']:.3f}, overshoot={h['overshoot']:.3f}, "
                f"grad_cam_threshold={h['grad_cam_threshold']:.2f}, "
                f"max_iter={h['max_iter']} → "
                f"fooled={h['fooled']}, SSIM={h['ssim']:.4f}, "
                f"new_prob={h['new_prob']:.4f}, iters_used={h['iters_used']}\n"
            )
        last = history[-1]
        if not last["fooled"]:
            user_msg += f"\nThe model was NOT fooled in round {last['round']}. Adjust parameters to improve.\n"
        else:
            user_msg += f"\nThe model WAS fooled in round {last['round']} but SSIM={last['ssim']:.4f} was below {SSIM_FLOOR}. Reduce aggression.\n"
    else:
        user_msg += "\nThis is the first round. Choose initial parameters based on the original confidence.\n"

    user_msg += "\nDecide the next hyperparameters. Return only JSON."

    # Delay to respect free tier rate limits
    time.sleep(1.0)

    # Generate response with retries for 429 server errors
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}
                ],
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content.strip()
            break
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                log.warning(f"Rate limit hit! Sleeping for 20s before retry {attempt+1}/3...")
                time.sleep(20)
            else:
                raise e

    # Parse JSON (strip markdown fences if present)
    raw_clean = raw.replace("```json", "").replace("```", "").strip()
    params = json.loads(raw_clean)

    # Safety clamp — agent must stay in allowed ranges
    params["epsilon"]          = float(np.clip(params.get("epsilon", 0.10), 0.01, 0.30))
    params["overshoot"]        = float(np.clip(params.get("overshoot", 1.02), 1.00, 1.20))
    params["grad_cam_threshold"] = float(np.clip(params.get("grad_cam_threshold", 0.4), 0.10, 0.60))
    params["max_iter"]         = int(np.clip(params.get("max_iter", 200), 50, 500))
    params["reasoning"]        = params.get("reasoning", "")

    return params


# ═════════════════════════════════════════════
#  SECTION 4 — Main Pipeline
# ═════════════════════════════════════════════

def main():
    # ── Sanity checks ──────────────────────────────
    for path, label in [(MODEL_PATH, "Model"), (INPUT_DIR, "Input directory")]:
        if not os.path.exists(path):
            log.error(f"{label} not found: {path}")
            return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── OpenAI client ──────────────────────────────
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        log.error("OPENAI_API_KEY environment variable not set. Please add it to your .env file.")
        return
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)

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

    # ── CSV + JSONL setup ──────────────────────────
    csv_cols = [
        "image", "original_label", "original_confidence",
        "adversarial_label", "adversarial_confidence",
        "total_rounds", "total_iterations", "fooled", "final_SSIM",
        "final_epsilon", "final_overshoot", "final_grad_cam_threshold"
    ]

    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f_csv, \
         open(AGENT_LOG, "w", encoding="utf-8") as f_jsonl:

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

                # ── Agent loop ─────────────────────────────
                history          = []
                total_iters      = 0
                final_result     = None
                final_params     = {}
                final_adv        = img_batch.copy()

                for rnd in range(1, MAX_ROUNDS + 1):
                    log.info(f"  ── Agent Round {rnd} ──")

                    # 1. Ask agent for hyperparameters
                    params = ask_agent(
                        client, img_name, orig_label, prob0, rnd, history
                    )
                    log.info(
                        f"  Agent → ε={params['epsilon']:.3f}, "
                        f"overshoot={params['overshoot']:.3f}, "
                        f"gcam_thresh={params['grad_cam_threshold']:.2f}, "
                        f"max_iter={params['max_iter']}"
                    )
                    log.info(f"  Reasoning: {params['reasoning']}")

                    # 2. Build Grad-CAM mask with agent-chosen threshold
                    if last_conv is not None:
                        mask, _ = make_gradcam_mask(
                            img_batch, model, orig_idx, last_conv,
                            threshold=params["grad_cam_threshold"]
                        )
                    else:
                        mask = np.ones((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

                    mask_pct = float(mask.mean()) * 100
                    log.info(f"  Grad-CAM mask coverage: {mask_pct:.1f}%")

                    # 3. Run DeepFool for this round
                    adv_batch, iters, adv_idx, adv_prob, ssim_val = deepfool_round(
                        model, img_batch, mask,
                        epsilon   = params["epsilon"],
                        overshoot = params["overshoot"],
                        max_iter  = params["max_iter"],
                    )

                    total_iters += iters
                    fooled       = (adv_idx != orig_idx)
                    adv_label    = TARGET_CLASSES[adv_idx]
                    adv_conf     = adv_prob if adv_idx == 1 else (1.0 - adv_prob)

                    log.info(
                        f"  Result → {adv_label} (sigmoid={adv_prob:.4f}) | "
                        f"Fooled={fooled} | SSIM={ssim_val:.4f} | Iters={iters}"
                    )

                    # 4. Record history for agent
                    round_record = {
                        "image": img_name,
                        "round": rnd,
                        "epsilon":            params["epsilon"],
                        "overshoot":          params["overshoot"],
                        "grad_cam_threshold": params["grad_cam_threshold"],
                        "max_iter":           params["max_iter"],
                        "reasoning":          params["reasoning"],
                        "fooled":             fooled,
                        "ssim":               ssim_val,
                        "new_prob":           adv_prob,
                        "iters_used":         iters,
                        "mask_coverage_pct":  round(mask_pct, 2),
                    }
                    history.append(round_record)
                    f_jsonl.write(json.dumps(round_record) + "\n")
                    f_jsonl.flush()

                    final_adv    = adv_batch
                    final_params = params
                    final_result = round_record

                    # ── SUCCESS: fooled AND SSIM >= floor ──
                    if fooled and ssim_val >= SSIM_FLOOR:
                        log.info(
                            f"  ✓ SUCCESS at round {rnd}! "
                            f"Fooled={fooled}, SSIM={ssim_val:.4f} ≥ {SSIM_FLOOR}"
                        )
                        break

                    # ── Fooled but SSIM too low → next round with smaller ε
                    if fooled and ssim_val < SSIM_FLOOR:
                        log.info(
                            f"  Fooled but SSIM={ssim_val:.4f} < {SSIM_FLOOR}. "
                            f"Agent will reduce aggression."
                        )
                        # Reset to original image so next round starts fresh
                        img_batch = np.expand_dims(img_np, 0).astype(np.float32)
                        continue

                # ── Save final adversarial image ───────────
                adv_uint8 = (np.clip(final_adv[0], 0, 1) * 255).astype(np.uint8)
                base_name = os.path.splitext(img_name)[0]
                out_path  = os.path.join(OUTPUT_DIR, f"{base_name}_agent_adv.png")
                cv2.imwrite(out_path, cv2.cvtColor(adv_uint8, cv2.COLOR_RGB2BGR))

                # ── CSV row ────────────────────────────────
                success = (
                    final_result["fooled"] and
                    final_result["ssim"] >= SSIM_FLOOR
                ) if final_result else False

                writer.writerow({
                    "image":                img_name,
                    "original_label":       orig_label,
                    "original_confidence":  f"{orig_conf:.4f}",
                    "adversarial_label":    TARGET_CLASSES[final_result["new_prob"] >= 0.5 and 1 or 0]
                                           if final_result else "N/A",
                    "adversarial_confidence": f"{final_result['new_prob']:.4f}"
                                              if final_result else "N/A",
                    "total_rounds":         len(history),
                    "total_iterations":     total_iters,
                    "fooled":               str(success),
                    "final_SSIM":           f"{final_result['ssim']:.4f}"
                                            if final_result else "N/A",
                    "final_epsilon":        f"{final_params.get('epsilon', 'N/A')}",
                    "final_overshoot":      f"{final_params.get('overshoot', 'N/A')}",
                    "final_grad_cam_threshold": f"{final_params.get('grad_cam_threshold', 'N/A')}",
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
                    "total_rounds": 0,
                    "total_iterations": 0,
                    "fooled": "False",
                    "final_SSIM": "N/A",
                    "final_epsilon": "N/A",
                    "final_overshoot": "N/A",
                    "final_grad_cam_threshold": "N/A",
                })
                f_csv.flush()

    log.info(f"\n✓ Done. Adversarial images → '{OUTPUT_DIR}/'")
    log.info(f"   Attack CSV   → '{LOG_FILE}'")
    log.info(f"   Agent decisions → '{AGENT_LOG}'")


if __name__ == "__main__":
    main()
