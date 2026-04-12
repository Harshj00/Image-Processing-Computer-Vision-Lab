# ============================================================
# Name         : Harsh Kumar Jha
# Course       : Image Processing & Computer Vision
# Assignment2   : Image Restoration for Surveillance Camera Systems


"""
Image Restoration for Surveillance Camera Systems
--------------------------------------------------
Simulates real-world surveillance noise (Gaussian + Salt-and-Pepper)
and restores image quality using classical spatial filters.

Per-task image outputs saved to outputs/:
  Task 1 → task1_surveillance_original.png
  Task 2 → task2_noisy_images.png
  Task 3 → task3_restored_images.png
  Task 4 → task4_performance_metrics.png  (bar charts)
  Task 5 → task5_filter_comparison.png
  Final  → master_comparison.png
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIG_BG   = "#0d1117"
TITLE_KW = dict(fontsize=11, color="white", fontweight="bold", pad=7)
ANNO_KW  = dict(fontsize=8,  color="#8b949e")


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

def save_fig(fig, filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   💾  Saved → {path}")
    return path


def compute_mse(original: np.ndarray, restored: np.ndarray) -> float:
    """Mean Squared Error — lower is better."""
    return np.mean((original.astype(np.float64) -
                    restored.astype(np.float64)) ** 2)


def compute_psnr(original: np.ndarray, restored: np.ndarray) -> float:
    """PSNR = 10·log10(255² / MSE) — higher is better."""
    mse = compute_mse(original, restored)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def fmt_psnr(val: float) -> str:
    return "∞" if val == float("inf") else f"{val:.2f} dB"


# ─────────────────────────────────────────────────────────────
# WELCOME
# ─────────────────────────────────────────────────────────────

def print_welcome():
    print("=" * 65)
    print("   IMAGE RESTORATION — SURVEILLANCE CAMERA SYSTEM")
    print("=" * 65)
    print("  Course : Image Processing & Computer Vision")
    print("  Unit   : Unit 2 — Noise Modeling & Image Restoration")
    print("  Author : Harsh")
    print("-" * 65)
    print("  Simulates Gaussian and Salt-and-Pepper noise that")
    print("  surveillance cameras experience in low-light, rain,")
    print("  fog, and dusty environments.")
    print("  Restores quality using Mean, Median, and Gaussian filters.")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────
# TASK 1 — IMAGE SELECTION & PREPROCESSING
# Output: task1_surveillance_original.png
# ─────────────────────────────────────────────────────────────

def load_image(image_path: str, target_size=(512, 512)):
    """
    Load a surveillance-style image, convert to grayscale.
    Saves task1_surveillance_original.png.
    Returns gray (np.ndarray).
    """
    print(f"\n[Task 1] Image Selection & Preprocessing")
    print(f"  Source: {image_path}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")

    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    print(f"  Resized to   : {target_size}")
    print(f"  Pixel range  : [{gray.min()}, {gray.max()}]")
    print(f"  Mean value   : {gray.mean():.1f}")

    # ── Task 1 figure ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Task 1 — Image Acquisition: Surveillance Scene",
                 fontsize=14, color="white", fontweight="bold")

    axes[0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original (Color)", **TITLE_KW)
    axes[0].set_xlabel(f"Size: {target_size[0]}×{target_size[1]}", **ANNO_KW)
    axes[0].axis("off")

    axes[1].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Grayscale (preprocessing)", **TITLE_KW)
    axes[1].set_xlabel(f"Mean intensity: {gray.mean():.1f}", **ANNO_KW)
    axes[1].axis("off")

    plt.tight_layout()
    save_fig(fig, "task1_surveillance_original.png")

    return gray


# ─────────────────────────────────────────────────────────────
# TASK 2 — NOISE MODELING
# Output: task2_noisy_images.png
# ─────────────────────────────────────────────────────────────

def add_gaussian_noise(gray: np.ndarray, mean=0, sigma=30) -> np.ndarray:
    """
    Add Gaussian noise ~ N(mean, sigma²) — models sensor/thermal noise
    in low-light surveillance cameras.
    """
    noise  = np.random.normal(mean, sigma, gray.shape)
    noisy  = np.clip(gray.astype(np.float64) + noise, 0, 255)
    return noisy.astype(np.uint8)


def add_salt_pepper_noise(gray: np.ndarray, salt_prob=0.04,
                           pepper_prob=0.04) -> np.ndarray:
    """
    Add Salt-and-Pepper noise — models transmission/bit errors.
    salt_prob  : fraction of pixels set to 255 (white)
    pepper_prob: fraction of pixels set to 0   (black)
    """
    noisy = gray.copy()
    total = gray.size

    # Salt (white)
    salt_coords = np.unravel_index(
        np.random.choice(total, int(salt_prob * total), replace=False),
        gray.shape)
    noisy[salt_coords] = 255

    # Pepper (black)
    pepper_coords = np.unravel_index(
        np.random.choice(total, int(pepper_prob * total), replace=False),
        gray.shape)
    noisy[pepper_coords] = 0

    return noisy


def model_noise(gray: np.ndarray):
    """
    Generate Gaussian and Salt-and-Pepper noisy versions.
    Saves task2_noisy_images.png.
    Returns (gauss_noisy, sp_noisy).
    """
    print("\n[Task 2] Noise Modeling")

    gauss_noisy = add_gaussian_noise(gray, mean=0, sigma=30)
    sp_noisy    = add_salt_pepper_noise(gray, salt_prob=0.04, pepper_prob=0.04)

    g_psnr = compute_psnr(gray, gauss_noisy)
    s_psnr = compute_psnr(gray, sp_noisy)
    g_mse  = compute_mse(gray, gauss_noisy)
    s_mse  = compute_mse(gray, sp_noisy)

    print(f"  Gaussian noise    — MSE: {g_mse:.2f}  PSNR: {fmt_psnr(g_psnr)}")
    print(f"  Salt-and-Pepper   — MSE: {s_mse:.2f}  PSNR: {fmt_psnr(s_psnr)}")

    # ── Task 2 figure ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Task 2 — Noise Modeling: Gaussian & Salt-and-Pepper",
                 fontsize=14, color="white", fontweight="bold")

    data = [
        ("Original (clean)",    gray,        "Baseline"),
        ("Gaussian Noise\n(σ=30, sensor/thermal)", gauss_noisy,
         f"PSNR: {fmt_psnr(g_psnr)} | MSE: {g_mse:.1f}"),
        ("Salt-and-Pepper\n(4% salt + 4% pepper)", sp_noisy,
         f"PSNR: {fmt_psnr(s_psnr)} | MSE: {s_mse:.1f}"),
    ]

    for ax, (title, img, note) in zip(axes, data):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, **TITLE_KW)
        ax.set_xlabel(note, **ANNO_KW)
        ax.axis("off")

    plt.tight_layout()
    save_fig(fig, "task2_noisy_images.png")

    return gauss_noisy, sp_noisy


# ─────────────────────────────────────────────────────────────
# TASK 3 — IMAGE RESTORATION TECHNIQUES
# Output: task3_restored_images.png
# ─────────────────────────────────────────────────────────────

def restore_images(gray: np.ndarray,
                   gauss_noisy: np.ndarray,
                   sp_noisy: np.ndarray):
    """
    Apply Mean, Median, and Gaussian filters to both noisy images.
    Saves task3_restored_images.png (2-row grid).
    Returns (gauss_restored, sp_restored) dicts.
    """
    print("\n[Task 3] Image Restoration — Spatial Filtering")

    kernel_size = (5, 5)
    sigma_gauss = 1.5

    def apply_filters(noisy, label):
        """Apply all three filters and return dict."""
        mean_f   = cv2.blur(noisy, kernel_size)
        median_f = cv2.medianBlur(noisy, 5)
        gauss_f  = cv2.GaussianBlur(noisy, kernel_size, sigma_gauss)

        results = {
            "Mean Filter":     mean_f,
            "Median Filter":   median_f,
            "Gaussian Filter": gauss_f,
        }

        print(f"  Filters on {label}:")
        for name, img in results.items():
            p = compute_psnr(gray, img)
            m = compute_mse(gray, img)
            print(f"    {name:18s} → PSNR: {fmt_psnr(p)}  MSE: {m:.2f}")

        return results

    gauss_restored = apply_filters(gauss_noisy, "Gaussian Noisy")
    sp_restored    = apply_filters(sp_noisy,    "Salt-and-Pepper Noisy")

    # ── Task 3 figure ─────────────────────────────────────────
    # 2 rows (one per noise type) × 4 cols (noisy + 3 filters)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Task 3 — Restoration: Mean / Median / Gaussian Filters",
                 fontsize=14, color="white", fontweight="bold")

    rows = [
        ("Gaussian Noisy",        gauss_noisy, gauss_restored),
        ("Salt-and-Pepper Noisy", sp_noisy,    sp_restored),
    ]

    for r, (noise_lbl, noisy_img, restored) in enumerate(rows):
        # Col 0: noisy input
        axes[r][0].imshow(noisy_img, cmap="gray", vmin=0, vmax=255)
        axes[r][0].set_title(f"Input:\n{noise_lbl}", **TITLE_KW)
        axes[r][0].set_xlabel(
            f"PSNR: {fmt_psnr(compute_psnr(gray, noisy_img))}", **ANNO_KW)
        axes[r][0].axis("off")

        for c, (name, img) in enumerate(restored.items(), start=1):
            p = compute_psnr(gray, img)
            axes[r][c].imshow(img, cmap="gray", vmin=0, vmax=255)
            axes[r][c].set_title(name, **TITLE_KW)
            axes[r][c].set_xlabel(f"PSNR: {fmt_psnr(p)}", **ANNO_KW)
            axes[r][c].axis("off")

    plt.tight_layout()
    save_fig(fig, "task3_restored_images.png")

    return gauss_restored, sp_restored


# ─────────────────────────────────────────────────────────────
# TASK 4 — PERFORMANCE EVALUATION
# Output: task4_performance_metrics.png
# ─────────────────────────────────────────────────────────────

def evaluate_performance(gray: np.ndarray,
                          gauss_noisy: np.ndarray, sp_noisy: np.ndarray,
                          gauss_restored: dict, sp_restored: dict):
    """
    Compute MSE + PSNR for all noisy and restored images.
    Print results table. Saves task4_performance_metrics.png (bar chart).
    """
    print("\n[Task 4] Performance Evaluation — MSE & PSNR")

    records = {}

    all_data = {
        "Gaussian Noisy":   gauss_noisy,
        "SP Noisy":         sp_noisy,
        **{f"G→{k}": v for k, v in gauss_restored.items()},
        **{f"SP→{k}": v for k, v in sp_restored.items()},
    }

    print(f"\n  {'Image':28s} {'MSE':>10}   {'PSNR':>12}")
    print(f"  {'-'*28} {'-'*10}   {'-'*12}")
    for name, img in all_data.items():
        mse  = compute_mse(gray, img)
        psnr = compute_psnr(gray, img)
        records[name] = (mse, psnr)
        ps = fmt_psnr(psnr)
        print(f"  {name:28s} {mse:10.2f}   {ps:>12}")

    # ── Task 4 figure: PSNR bar chart ─────────────────────────
    labels  = list(records.keys())
    psnr_vals = [v[1] if v[1] != float("inf") else 99 for v in records.values()]
    mse_vals  = [v[0] for v in records.values()]

    colors = (["#ff6b6b", "#ff9f43"] +          # noisy
              ["#4cc9f0"] * 3 +                 # gauss restored
              ["#f9c74f"] * 3)                  # sp restored

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Task 4 — Performance Evaluation: MSE & PSNR",
                 fontsize=14, color="white", fontweight="bold")

    for ax, vals, ylabel, title, good_is_high in [
        (ax1, psnr_vals, "PSNR (dB)",
         "PSNR — higher is better", True),
        (ax2, mse_vals,  "MSE",
         "MSE  — lower is better",  False),
    ]:
        ax.set_facecolor("#0d1117")
        bars = ax.bar(labels, vals, color=colors,
                      edgecolor="white", linewidth=0.6)
        ax.set_title(title, **TITLE_KW)
        ax.set_ylabel(ylabel, color="white")
        ax.tick_params(colors="white", labelsize=7.5)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_ylim(0, max(vals) * 1.22)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.1f}", ha="center",
                    color="white", fontsize=7.5, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    plt.tight_layout()
    save_fig(fig, "task4_performance_metrics.png")

    return records


# ─────────────────────────────────────────────────────────────
# TASK 5 — ANALYTICAL DISCUSSION
# Output: task5_filter_comparison.png
# ─────────────────────────────────────────────────────────────

def analytical_discussion(gray, gauss_noisy, sp_noisy,
                           gauss_restored, sp_restored):
    """
    Print filter-wise comparison and justify best method per noise type.
    Saves task5_filter_comparison.png.
    """
    print("\n" + "=" * 65)
    print("[Task 5] Analytical Discussion")
    print("=" * 65)

    # Best filter per noise type
    best_gauss = max(gauss_restored.items(),
                     key=lambda kv: compute_psnr(gray, kv[1]))
    best_sp    = max(sp_restored.items(),
                     key=lambda kv: compute_psnr(gray, kv[1]))

    print(f"\n  Best filter for Gaussian noise   → {best_gauss[0]}")
    print(f"  Best filter for Salt-and-Pepper  → {best_sp[0]}")

    print("""
  Filter-Wise Analysis
  ─────────────────────────────────────────────────────────
  Mean Filter:
    · Averages a k×k neighbourhood.
    · Reduces Gaussian noise well (averaging cancels random errors).
    · Blurs edges — bad for scenes where license-plate / face
      sharpness is needed.
    · Performs poorly on S&P noise: isolated bright/dark pixels
      get spread across the neighbourhood.

  Median Filter:
    · Replaces each pixel with the median of its neighbourhood.
    · Excellent for Salt-and-Pepper noise: impulse values are
      statistical outliers and the median ignores them.
    · Better edge preservation than mean filtering.
    · Less effective on Gaussian noise (median is non-linear;
      it does not average out smooth Gaussian fluctuations).

  Gaussian Filter:
    · Applies a weighted average where central pixels contribute more.
    · Smoother than mean filter — preserves more local structure.
    · Good for Gaussian noise; moderate edge blurring.
    · Like mean filter, spreads S&P impulses rather than removes them.

  Recommendation for Surveillance Systems:
    · Gaussian / sensor noise  → Gaussian or Mean filter.
    · Transmission / S&P noise → Median filter (clear winner).
    · Combined noise           → Median first, then Gaussian pass.
""")

    # ── Task 5 figure: side-by-side best filters ──────────────
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Task 5 — Filter Comparison: Best Restoration per Noise Type",
                 fontsize=14, color="white", fontweight="bold")

    row_data = [
        ("Gaussian Noise", gauss_noisy, gauss_restored),
        ("Salt-and-Pepper", sp_noisy,  sp_restored),
    ]

    for r, (noise_label, noisy, restored) in enumerate(row_data):
        # Original
        axes[r][0].imshow(gray, cmap="gray", vmin=0, vmax=255)
        axes[r][0].set_title("Original (clean)", **TITLE_KW)
        axes[r][0].axis("off")

        # Noisy
        p_noisy = compute_psnr(gray, noisy)
        axes[r][1].imshow(noisy, cmap="gray", vmin=0, vmax=255)
        axes[r][1].set_title(f"Noisy\n({noise_label})", **TITLE_KW)
        axes[r][1].set_xlabel(f"PSNR: {fmt_psnr(p_noisy)}", **ANNO_KW)
        axes[r][1].axis("off")

        # Best filter
        best_name, best_img = max(restored.items(),
                                  key=lambda kv: compute_psnr(gray, kv[1]))
        p_best = compute_psnr(gray, best_img)
        axes[r][2].imshow(best_img, cmap="gray", vmin=0, vmax=255)
        axes[r][2].set_title(f"Best Filter: {best_name}", **TITLE_KW)
        axes[r][2].set_xlabel(
            f"PSNR: {fmt_psnr(p_best)}  ← improvement from {fmt_psnr(p_noisy)}",
            **ANNO_KW)
        axes[r][2].axis("off")

    plt.tight_layout()
    save_fig(fig, "task5_filter_comparison.png")


# ─────────────────────────────────────────────────────────────
# MASTER COMPARISON — all stages
# Output: master_comparison.png
# ─────────────────────────────────────────────────────────────

def master_comparison(gray, gauss_noisy, sp_noisy,
                       gauss_restored, sp_restored):
    """
    Large summary figure: clean | both noisy | all 6 restored images.
    """
    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Assignment 2 — Surveillance Restoration: Full Pipeline",
                 fontsize=16, color="white", fontweight="bold")

    # Row 0: original + both noisy + blank
    items_r0 = [
        ("Original (clean)", gray),
        ("Gaussian Noisy",   gauss_noisy),
        ("Salt-and-Pepper",  sp_noisy),
    ]
    for c, (title, img) in enumerate(items_r0):
        p = compute_psnr(gray, img)
        axes[0][c].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[0][c].set_title(f"Task 1-2 · {title}", **TITLE_KW)
        axes[0][c].set_xlabel(
            "Baseline" if img is gray else f"PSNR: {fmt_psnr(p)}", **ANNO_KW)
        axes[0][c].axis("off")
    axes[0][3].axis("off")

    # Row 1: Gaussian noise restored
    for c, (name, img) in enumerate(gauss_restored.items()):
        p = compute_psnr(gray, img)
        axes[1][c].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[1][c].set_title(f"Task 3 · G-Noise\n{name}", **TITLE_KW)
        axes[1][c].set_xlabel(f"PSNR: {fmt_psnr(p)}", **ANNO_KW)
        axes[1][c].axis("off")
    axes[1][3].axis("off")
    axes[1][3].text(0.5, 0.5,
                    "Gaussian Noise\nRestoration",
                    transform=axes[1][3].transAxes,
                    ha="center", va="center",
                    color="#4cc9f0", fontsize=13, style="italic")

    # Row 2: S&P noise restored
    for c, (name, img) in enumerate(sp_restored.items()):
        p = compute_psnr(gray, img)
        axes[2][c].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[2][c].set_title(f"Task 3 · S&P Noise\n{name}", **TITLE_KW)
        axes[2][c].set_xlabel(f"PSNR: {fmt_psnr(p)}", **ANNO_KW)
        axes[2][c].axis("off")
    axes[2][3].axis("off")
    axes[2][3].text(0.5, 0.5,
                    "S&P Noise\nRestoration",
                    transform=axes[2][3].transAxes,
                    ha="center", va="center",
                    color="#f9c74f", fontsize=13, style="italic")

    plt.tight_layout()
    save_fig(fig, "master_comparison.png")


# ─────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────

def process_image(image_path: str):
    print(f"\n{'━'*65}\n  Processing: {image_path}\n{'━'*65}")

    gray                         = load_image(image_path)          # Task 1
    gauss_noisy, sp_noisy        = model_noise(gray)               # Task 2
    gauss_restored, sp_restored  = restore_images(gray,            # Task 3
                                       gauss_noisy, sp_noisy)
    evaluate_performance(gray,                                      # Task 4
        gauss_noisy, sp_noisy, gauss_restored, sp_restored)
    analytical_discussion(gray,                                     # Task 5
        gauss_noisy, sp_noisy, gauss_restored, sp_restored)
    master_comparison(gray,                                         # Master
        gauss_noisy, sp_noisy, gauss_restored, sp_restored)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    print_welcome()

    default_images = [
        "sample_images/street_view.jpg",
        "sample_images/parking_area.jpg",
        "sample_images/corridor.jpg",
    ]
    image_paths = sys.argv[1:] if len(sys.argv) > 1 else default_images
    if not sys.argv[1:]:
        print("ℹ  No CLI args — using default_images list.\n"
              "   Run: python restoration.py img1.jpg img2.jpg img3.jpg\n")

    for idx, path in enumerate(image_paths, 1):
        print(f"\n{'='*65}\n  SAMPLE RUN {idx} / {len(image_paths)}\n{'='*65}")
        try:
            process_image(path)
        except (FileNotFoundError, ValueError) as e:
            print(f"  ⚠  Skipping — {e}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n{'='*65}")
    print("  ✅  All runs complete!")
    print(f"  📁  Outputs: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'='*65}")
    print("\n  Files generated per run:")
    print("    • task1_surveillance_original.png")
    print("    • task2_noisy_images.png")
    print("    • task3_restored_images.png")
    print("    • task4_performance_metrics.png")
    print("    • task5_filter_comparison.png")
    print("    • master_comparison.png\n")


if __name__ == "__main__":
    main()
