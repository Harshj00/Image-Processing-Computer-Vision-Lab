# ============================================================
# Name         : Harsh
# Course       : Image Processing & Computer Vision
# Assignment   : Intelligent Image Enhancement & Analysis System
#                (Capstone — Units 1–4)
# Date         : 2026
# ============================================================

"""
Intelligent Image Enhancement & Analysis System
-------------------------------------------------
End-to-end image processing pipeline integrating all course units.

Unit 1 → Acquisition, grayscale conversion, sampling awareness
Unit 2 → Noise modeling, filtering, enhancement (CLAHE)
Unit 3 → Segmentation (global, Otsu), morphological refinement
Unit 4 → Edge detection, contours, ORB feature extraction
Capstone → PSNR, MSE, SSIM evaluation + full pipeline visualisation

Per-task image outputs (saved to outputs/):
  Task 1 → (project setup — no image output)
  Task 2 → task2_acquisition.png
  Task 3 → task3_enhancement_restoration.png
  Task 4 → task4_segmentation_morphology.png
  Task 5 → task5_object_features.png
  Task 6 → task6_performance_metrics.png
  Task 7 → task7_final_pipeline.png
  Final  → master_comparison.png
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

# Try to import SSIM; fall back gracefully if skimage not available
try:
    from skimage.metrics import structural_similarity as ssim_func
    HAVE_SSIM = True
except ImportError:
    HAVE_SSIM = False
    print("  ⚠  scikit-image not found — SSIM will be skipped.")
    print("     Install with: pip install scikit-image")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIG_BG   = "#0b0c1a"
TITLE_KW = dict(fontsize=11, color="white", fontweight="bold", pad=7)
ANNO_KW  = dict(fontsize=8,  color="#9ec5fe")


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


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    mse = compute_mse(original, modified)
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def compute_ssim(original: np.ndarray, modified: np.ndarray) -> float:
    if not HAVE_SSIM:
        return -1.0
    # ssim_func expects same dtype and range
    return float(ssim_func(original, modified, data_range=255))


def fmt_psnr(v: float) -> str:
    return "∞" if v == float("inf") else f"{v:.2f} dB"


def fmt_ssim(v: float) -> str:
    return "N/A" if v < 0 else f"{v:.4f}"


# ─────────────────────────────────────────────────────────────
# TASK 1 — PROJECT SETUP & SYSTEM OVERVIEW
# ─────────────────────────────────────────────────────────────

def print_welcome():
    print("=" * 65)
    print("   INTELLIGENT IMAGE ENHANCEMENT & ANALYSIS SYSTEM")
    print("   (Capstone Project — All Units)")
    print("=" * 65)
    print("  Course  : Image Processing & Computer Vision")
    print("  Author  : Harsh")
    print("-" * 65)
    print("  End-to-end pipeline:")
    print("  [1] Acquisition & preprocessing       (Unit 1)")
    print("  [2] Noise modeling & restoration       (Unit 2)")
    print("  [3] Segmentation & morphology          (Unit 3)")
    print("  [4] Edge detection & feature extract   (Unit 4)")
    print("  [5] PSNR / MSE / SSIM evaluation")
    print("  [6] Full pipeline visualisation")
    print("  Every stage saves its own output image.")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────
# TASK 2 — IMAGE ACQUISITION & PREPROCESSING
# Output: task2_acquisition.png
# ─────────────────────────────────────────────────────────────

def acquire_image(image_path: str, target_size=(512, 512)):
    """
    Load image → resize → grayscale.
    Saves task2_acquisition.png.
    Returns (bgr, gray).
    """
    print(f"\n[Task 2] Image Acquisition & Preprocessing")
    print(f"  Source: {image_path}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")

    bgr  = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    print(f"  Original dims  : {img.shape[1]}×{img.shape[0]} px")
    print(f"  Standardised   : {target_size[0]}×{target_size[1]} px")
    print(f"  Pixel range    : [{gray.min()}, {gray.max()}]")
    print(f"  Mean intensity : {gray.mean():.1f}")

    # ── figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Task 2 — Image Acquisition & Preprocessing",
                 fontsize=14, color="white", fontweight="bold")

    axes[0].imshow(bgr_to_rgb(bgr))
    axes[0].set_title("Original (Color, 512×512)", **TITLE_KW)
    axes[0].set_xlabel("Loaded from disk, resized", **ANNO_KW)
    axes[0].axis("off")

    axes[1].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Grayscale (8-bit)", **TITLE_KW)
    axes[1].set_xlabel(f"Mean: {gray.mean():.1f}  |  Std: {gray.std():.1f}",
                       **ANNO_KW)
    axes[1].axis("off")

    plt.tight_layout()
    save_fig(fig, "task2_acquisition.png")

    return bgr, gray


# ─────────────────────────────────────────────────────────────
# TASK 3 — IMAGE ENHANCEMENT & RESTORATION
# Output: task3_enhancement_restoration.png
# ─────────────────────────────────────────────────────────────

def enhance_and_restore(gray: np.ndarray):
    """
    Add Gaussian + S&P noise → restore with Mean/Median/Gaussian filters
    → enhance contrast with CLAHE.
    Saves task3_enhancement_restoration.png.
    Returns (gauss_noisy, sp_noisy, gauss_restored, sp_restored,
             clahe_enhanced).
    """
    print("\n[Task 3] Image Enhancement & Restoration")

    # ── Add noise ─────────────────────────────────────────────
    noise   = np.random.normal(0, 25, gray.shape)
    g_noisy = np.clip(gray.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    sp_noisy = gray.copy()
    total    = gray.size
    for val, prob in [(255, 0.03), (0, 0.03)]:
        idx = np.unravel_index(
            np.random.choice(total, int(prob * total), replace=False),
            gray.shape)
        sp_noisy[idx] = val

    # ── Restore ───────────────────────────────────────────────
    k = (5, 5)
    gauss_restored = {
        "Mean":     cv2.blur(g_noisy, k),
        "Median":   cv2.medianBlur(g_noisy, 5),
        "Gaussian": cv2.GaussianBlur(g_noisy, k, 1.5),
    }
    sp_restored = {
        "Mean":     cv2.blur(sp_noisy, k),
        "Median":   cv2.medianBlur(sp_noisy, 5),
        "Gaussian": cv2.GaussianBlur(sp_noisy, k, 1.5),
    }

    # ── CLAHE contrast enhancement ────────────────────────────
    clahe         = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    best_restored = gauss_restored["Gaussian"]
    clahe_img     = clahe.apply(best_restored)

    print(f"  Gaussian noise → best restore: Gaussian filter  "
          f"PSNR: {fmt_psnr(compute_psnr(gray, gauss_restored['Gaussian']))}")
    print(f"  S&P noise      → best restore: Median filter    "
          f"PSNR: {fmt_psnr(compute_psnr(gray, sp_restored['Median']))}")
    print(f"  CLAHE enhanced  PSNR: {fmt_psnr(compute_psnr(gray, clahe_img))}")

    # ── figure: 3 rows ─────────────────────────────────────────
    # Row 0: original | G-noisy | SP-noisy
    # Row 1: G-restored (3 filters)
    # Row 2: SP-restored (3 filters) + CLAHE
    fig, axes = plt.subplots(3, 4, figsize=(22, 13))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Task 3 — Enhancement & Restoration Pipeline",
                 fontsize=14, color="white", fontweight="bold")

    # Row 0
    for c, (title, img, note) in enumerate([
        ("Original",           gray,    "Clean baseline"),
        ("+ Gaussian Noise",   g_noisy, f"PSNR: {fmt_psnr(compute_psnr(gray,g_noisy))}"),
        ("+ Salt-Pepper Noise",sp_noisy,f"PSNR: {fmt_psnr(compute_psnr(gray,sp_noisy))}"),
        ("CLAHE Enhanced",     clahe_img,f"PSNR: {fmt_psnr(compute_psnr(gray,clahe_img))}"),
    ]):
        axes[0][c].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[0][c].set_title(title, **TITLE_KW)
        axes[0][c].set_xlabel(note, **ANNO_KW)
        axes[0][c].axis("off")

    # Row 1: Gaussian noise restored
    for c, (fname, fimg) in enumerate(gauss_restored.items()):
        p = compute_psnr(gray, fimg)
        axes[1][c].imshow(fimg, cmap="gray", vmin=0, vmax=255)
        axes[1][c].set_title(f"G-Noise → {fname} Filter", **TITLE_KW)
        axes[1][c].set_xlabel(f"PSNR: {fmt_psnr(p)}", **ANNO_KW)
        axes[1][c].axis("off")
    axes[1][3].axis("off")
    axes[1][3].text(0.5, 0.5, "Gaussian\nNoise\nRestoration",
                    transform=axes[1][3].transAxes, ha="center", va="center",
                    color="#4cc9f0", fontsize=13, style="italic")

    # Row 2: S&P noise restored
    for c, (fname, fimg) in enumerate(sp_restored.items()):
        p = compute_psnr(gray, fimg)
        axes[2][c].imshow(fimg, cmap="gray", vmin=0, vmax=255)
        axes[2][c].set_title(f"S&P → {fname} Filter", **TITLE_KW)
        axes[2][c].set_xlabel(f"PSNR: {fmt_psnr(p)}", **ANNO_KW)
        axes[2][c].axis("off")
    axes[2][3].axis("off")
    axes[2][3].text(0.5, 0.5, "S&P Noise\nRestoration",
                    transform=axes[2][3].transAxes, ha="center", va="center",
                    color="#f9c74f", fontsize=13, style="italic")

    plt.tight_layout()
    save_fig(fig, "task3_enhancement_restoration.png")

    return g_noisy, sp_noisy, gauss_restored, sp_restored, clahe_img


# ─────────────────────────────────────────────────────────────
# TASK 4 — IMAGE SEGMENTATION & MORPHOLOGICAL PROCESSING
# Output: task4_segmentation_morphology.png
# ─────────────────────────────────────────────────────────────

def segment_and_morph(gray: np.ndarray, enhanced: np.ndarray):
    """
    Global + Otsu thresholding on enhanced image → dilation + erosion.
    Saves task4_segmentation_morphology.png.
    Returns (global_seg, otsu_seg, dilated, eroded).
    """
    print("\n[Task 4] Segmentation & Morphological Processing")

    _, global_seg = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
    otsu_t, otsu_seg = cv2.threshold(enhanced, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(otsu_seg, kernel, iterations=2)
    eroded  = cv2.erode(otsu_seg,  kernel, iterations=2)
    opened  = cv2.morphologyEx(otsu_seg, cv2.MORPH_OPEN,  kernel)
    closed  = cv2.morphologyEx(otsu_seg, cv2.MORPH_CLOSE, kernel)

    print(f"  Otsu threshold        : T = {otsu_t:.1f}")
    print(f"  Foreground (Otsu)     : {100*np.count_nonzero(otsu_seg)/otsu_seg.size:.1f}%")
    print(f"  Foreground (Dilated)  : {100*np.count_nonzero(dilated)/dilated.size:.1f}%")
    print(f"  Foreground (Eroded)   : {100*np.count_nonzero(eroded)/eroded.size:.1f}%")

    # ── figure: 2 rows ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Task 4 — Segmentation & Morphological Processing",
                 fontsize=14, color="white", fontweight="bold")

    panels = [
        # Row 0
        [("Enhanced Image",       enhanced,   "CLAHE output"),
         ("Global Threshold\nT=127", global_seg, f"FG: {100*np.count_nonzero(global_seg)/global_seg.size:.1f}%"),
         (f"Otsu Threshold\nT={otsu_t:.0f}",  otsu_seg,  f"FG: {100*np.count_nonzero(otsu_seg)/otsu_seg.size:.1f}%"),
         ("Dilation\n(Otsu)",     dilated,    f"FG: {100*np.count_nonzero(dilated)/dilated.size:.1f}%")],
        # Row 1
        [("Erosion\n(Otsu)",      eroded,     f"FG: {100*np.count_nonzero(eroded)/eroded.size:.1f}%"),
         ("Opening\n(Otsu)",      opened,     f"FG: {100*np.count_nonzero(opened)/opened.size:.1f}%"),
         ("Closing\n(Otsu)",      closed,     f"FG: {100*np.count_nonzero(closed)/closed.size:.1f}%"),
         ("Original Gray",        gray,       "Reference")],
    ]

    for r, row in enumerate(panels):
        for c, (title, img, note) in enumerate(row):
            axes[r][c].imshow(img, cmap="gray", vmin=0, vmax=255)
            axes[r][c].set_title(title, **TITLE_KW)
            axes[r][c].set_xlabel(note, **ANNO_KW)
            axes[r][c].axis("off")

    plt.tight_layout()
    save_fig(fig, "task4_segmentation_morphology.png")

    return global_seg, otsu_seg, dilated, eroded


# ─────────────────────────────────────────────────────────────
# TASK 5 — OBJECT REPRESENTATION & FEATURE EXTRACTION
# Output: task5_object_features.png
# ─────────────────────────────────────────────────────────────

def object_and_features(bgr: np.ndarray, gray: np.ndarray,
                          otsu_seg: np.ndarray):
    """
    Sobel + Canny → contours + bboxes → ORB keypoints.
    Saves task5_object_features.png.
    Returns (canny, contour_img, bbox_img, keypoints).
    """
    print("\n[Task 5] Object Representation & Feature Extraction")

    # ── Edge detection ────────────────────────────────────────
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag   = np.sqrt(sobelx**2 + sobely**2)
    sobel_edges = np.uint8(np.clip(sobel_mag / (sobel_mag.max() + 1e-8) * 255, 0, 255))
    canny_edges = cv2.Canny(gray, 50, 150)

    # ── Contours & bounding boxes ─────────────────────────────
    contours, _ = cv2.findContours(
        otsu_seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= 300]

    contour_img = bgr.copy()
    bbox_img    = bgr.copy()
    cv2.drawContours(contour_img, valid, -1, (0, 255, 100), 2)

    areas, peris = [], []
    for cnt in valid:
        x, y, w, h = cv2.boundingRect(cnt)
        a = cv2.contourArea(cnt)
        p = cv2.arcLength(cnt, True)
        areas.append(a)
        peris.append(p)
        cv2.rectangle(bbox_img, (x, y), (x+w, y+h), (255, 80, 0), 2)
        cv2.putText(bbox_img, f"{int(a)}", (x, max(y-5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # ── ORB features ──────────────────────────────────────────
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    kp_img = cv2.drawKeypoints(bgr, keypoints, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    print(f"  Contours (≥300 px²)   : {len(valid)}")
    print(f"  Mean area             : {np.mean(areas) if areas else 0:.0f} px²")
    print(f"  ORB keypoints         : {len(keypoints)}")

    # ── figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Task 5 — Object Representation & Feature Extraction",
                 fontsize=14, color="white", fontweight="bold")

    panels = [
        ("Sobel Edges",      sobel_edges,  False, "hot",  "Gradient magnitude"),
        ("Canny Edges",      canny_edges,  False, "gray", "Binary precise edges"),
        ("Contours",         contour_img,  True,  None,   f"{len(valid)} objects"),
        ("Bounding Boxes",   bbox_img,     True,  None,   f"Mean area: {np.mean(areas) if areas else 0:.0f} px²"),
        ("ORB Keypoints",    kp_img,       True,  None,   f"{len(keypoints)} keypoints"),
        ("Original",         bgr,          True,  None,   "Reference"),
    ]

    for idx, (title, img, is_color, cmap, note) in enumerate(panels):
        r, c = divmod(idx, 3)
        if is_color:
            axes[r][c].imshow(bgr_to_rgb(img))
        else:
            axes[r][c].imshow(img, cmap=cmap)
        axes[r][c].set_title(title, **TITLE_KW)
        axes[r][c].set_xlabel(note, **ANNO_KW)
        axes[r][c].axis("off")

    plt.tight_layout()
    save_fig(fig, "task5_object_features.png")

    return canny_edges, contour_img, bbox_img, keypoints


# ─────────────────────────────────────────────────────────────
# TASK 6 — PERFORMANCE EVALUATION
# Output: task6_performance_metrics.png
# ─────────────────────────────────────────────────────────────

def evaluate_performance(gray, g_noisy, sp_noisy,
                          gauss_restored, sp_restored, clahe_img):
    """
    Compute MSE, PSNR, SSIM for all processing stages.
    Print table. Saves task6_performance_metrics.png.
    """
    print("\n" + "=" * 65)
    print("[Task 6] Performance Evaluation — MSE / PSNR / SSIM")
    print("=" * 65)

    candidates = {
        "Gaussian Noisy":         g_noisy,
        "S&P Noisy":              sp_noisy,
        "G-Restore Mean":         gauss_restored["Mean"],
        "G-Restore Median":       gauss_restored["Median"],
        "G-Restore Gaussian":     gauss_restored["Gaussian"],
        "SP-Restore Mean":        sp_restored["Mean"],
        "SP-Restore Median":      sp_restored["Median"],
        "SP-Restore Gaussian":    sp_restored["Gaussian"],
        "CLAHE Enhanced":         clahe_img,
    }

    records = {}
    print(f"\n  {'Stage':28s} {'MSE':>9} {'PSNR':>12} {'SSIM':>9}")
    print(f"  {'-'*28} {'-'*9} {'-'*12} {'-'*9}")

    for name, img in candidates.items():
        mse  = compute_mse(gray, img)
        psnr = compute_psnr(gray, img)
        ssim = compute_ssim(gray, img)
        records[name] = (mse, psnr, ssim)
        print(f"  {name:28s} {mse:9.2f} {fmt_psnr(psnr):>12} {fmt_ssim(ssim):>9}")

    # ── figure: triple bar chart ──────────────────────────────
    labels = list(records.keys())
    mses   = [v[0] for v in records.values()]
    psnrs  = [v[1] if v[1] != float("inf") else 99 for v in records.values()]
    ssims  = [v[2] for v in records.values()]

    colors = (["#ff6b6b", "#ff9f43"] +
              ["#4cc9f0"] * 3 +
              ["#a8dadc"] * 3 +
              ["#06d6a0"])

    fig, axes = plt.subplots(3, 1, figsize=(16, 13))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Task 6 — Performance Evaluation: MSE / PSNR / SSIM",
                 fontsize=14, color="white", fontweight="bold")

    for ax, vals, ylabel, title in [
        (axes[0], psnrs, "PSNR (dB)",    "PSNR — Peak Signal-to-Noise Ratio (higher = better)"),
        (axes[1], mses,  "MSE",          "MSE — Mean Squared Error (lower = better)"),
        (axes[2], ssims, "SSIM (0–1)",   "SSIM — Structural Similarity Index (higher = better)"),
    ]:
        if all(v < 0 for v in vals):   # SSIM unavailable
            ax.text(0.5, 0.5, "SSIM unavailable\n(install scikit-image)",
                    transform=ax.transAxes, ha="center", va="center",
                    color="white", fontsize=12)
            ax.axis("off")
            continue

        ax.set_facecolor("#10102a")
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
                    f"{val:.2f}", ha="center",
                    color="white", fontsize=7.5, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor("#20204a")

    plt.tight_layout()
    save_fig(fig, "task6_performance_metrics.png")

    return records


# ─────────────────────────────────────────────────────────────
# TASK 7 — FINAL VISUALISATION & ANALYSIS
# Output: task7_final_pipeline.png
# ─────────────────────────────────────────────────────────────

def final_pipeline_figure(bgr, gray, g_noisy, sp_noisy,
                           gauss_restored, clahe_img,
                           otsu_seg, canny_edges, kp_img,
                           perf_records):
    """
    Display all major stages in a single multi-panel figure.
    Print conclusion. Saves task7_final_pipeline.png.
    """
    print("\n[Task 7] Final Visualisation & System Conclusion")

    # ── print conclusion ──────────────────────────────────────
    best_psnr_name = max(
        ((k, v[1]) for k, v in perf_records.items() if v[1] != float("inf")),
        key=lambda x: x[1])[0]

    print(f"""
  System Performance Summary
  ─────────────────────────────────────────────────────────
  Best restoration PSNR  : {best_psnr_name}
    PSNR = {fmt_psnr(perf_records[best_psnr_name][1])}
    SSIM = {fmt_ssim(perf_records[best_psnr_name][2])}

  Pipeline Stages Completed
  1. Acquisition     : Loaded, resized to 512×512, grayscaled.
  2. Noise added     : Gaussian (σ=25) and Salt-Pepper (3% each).
  3. Restoration     : Mean / Median / Gaussian filters applied.
  4. Enhancement     : CLAHE contrast improvement applied.
  5. Segmentation    : Global T=127 and Otsu's auto-threshold.
  6. Morphology      : Dilation, Erosion, Opening, Closing.
  7. Edge Detection  : Sobel gradient map + Canny binary edges.
  8. Feature Extract : ORB keypoints and 32-byte descriptors.
  9. Evaluation      : MSE, PSNR, SSIM computed per stage.

  Conclusions
  · Median filter is best for impulse (S&P) noise.
  · Gaussian filter best balances noise suppression / sharpness.
  · CLAHE significantly improves local contrast without clipping.
  · Otsu's threshold adapts to image histogram — superior to fixed T.
  · Morphological closing best for filling segmented region holes.
  · Canny provides the most precise edges for object detection.
  · ORB descriptors enable real-time feature matching and tracking.
""")

    # ── figure: 2 rows × 4 panels ─────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Task 7 — Intelligent Image Processing: Full Pipeline Summary",
                 fontsize=15, color="white", fontweight="bold")

    panels = [
        # Row 0
        ("Original",         bgr,                        True,  None,   "Task 2: Acquisition"),
        ("Gaussian Noisy",   g_noisy,                    False, "gray", "Task 3: Degradation"),
        ("Best Restored",    gauss_restored["Gaussian"], False, "gray", "Task 3: Restoration"),
        ("CLAHE Enhanced",   clahe_img,                  False, "gray", "Task 3: Enhancement"),
        # Row 1
        ("Otsu Segmented",   otsu_seg,                   False, "gray", "Task 4: Segmentation"),
        ("Canny Edges",      canny_edges,                False, "gray", "Task 5: Edge Detection"),
        ("ORB Features",     kp_img,                     True,  None,   "Task 5: Feature Extract"),
        ("Grayscale Input",  gray,                       False, "gray", "Task 2: Preprocessing"),
    ]

    for idx, (title, img, is_color, cmap, note) in enumerate(panels):
        r, c = divmod(idx, 4)
        if is_color:
            axes[r][c].imshow(bgr_to_rgb(img))
        else:
            axes[r][c].imshow(img, cmap=cmap, vmin=0, vmax=255)
        axes[r][c].set_title(title, **TITLE_KW)
        axes[r][c].set_xlabel(note, **ANNO_KW)
        axes[r][c].axis("off")

    plt.tight_layout()
    save_fig(fig, "task7_final_pipeline.png")


# ─────────────────────────────────────────────────────────────
# MASTER COMPARISON
# Output: master_comparison.png
# ─────────────────────────────────────────────────────────────

def master_comparison(bgr, gray, g_noisy, sp_noisy,
                       gauss_restored, sp_restored, clahe_img,
                       global_seg, otsu_seg, dilated, eroded,
                       canny_edges, kp_img, keypoints):
    """Comprehensive overview — all pipeline stages."""
    fig = plt.figure(figsize=(28, 16))
    fig.patch.set_facecolor(FIG_BG)
    gs  = gridspec.GridSpec(4, 6, figure=fig, hspace=0.42, wspace=0.25)

    def add(ax, img, title, note, is_color=False, cmap="gray"):
        if is_color:
            ax.imshow(bgr_to_rgb(img))
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title, fontsize=9, color="white",
                     fontweight="bold", pad=5)
        ax.set_xlabel(note, fontsize=7, color="#9ec5fe")
        ax.axis("off")

    # Row 0 — Task 2
    add(fig.add_subplot(gs[0, 0:2]), bgr,    "T2·Original",  "Color input",    True)
    add(fig.add_subplot(gs[0, 2:4]), gray,   "T2·Grayscale", "Preprocessing")
    add(fig.add_subplot(gs[0, 4]),   g_noisy,"T3·G-Noisy",   "Gaussian σ=25")
    add(fig.add_subplot(gs[0, 5]),   sp_noisy,"T3·SP-Noisy", "S&P 3%")

    # Row 1 — Task 3
    add(fig.add_subplot(gs[1, 0]), gauss_restored["Mean"],    "T3·G→Mean",    "Restored")
    add(fig.add_subplot(gs[1, 1]), gauss_restored["Median"],  "T3·G→Median",  "Restored")
    add(fig.add_subplot(gs[1, 2]), gauss_restored["Gaussian"],"T3·G→Gaussian","Restored")
    add(fig.add_subplot(gs[1, 3]), sp_restored["Median"],     "T3·SP→Median", "Best for SP")
    add(fig.add_subplot(gs[1, 4]), clahe_img,                 "T3·CLAHE",     "Enhanced")
    ax_blank = fig.add_subplot(gs[1, 5])
    ax_blank.axis("off")

    # Row 2 — Task 4
    add(fig.add_subplot(gs[2, 0]), global_seg,"T4·Global Seg","T=127")
    add(fig.add_subplot(gs[2, 1]), otsu_seg,  "T4·Otsu Seg", "Auto-T")
    add(fig.add_subplot(gs[2, 2]), dilated,   "T4·Dilation", "ROI expand")
    add(fig.add_subplot(gs[2, 3]), eroded,    "T4·Erosion",  "ROI shrink")
    ax_blank2 = fig.add_subplot(gs[2, 4:6])
    ax_blank2.axis("off")
    ax_blank2.text(0.5, 0.5,
                   f"Task 5\nORB: {len(keypoints)} keypoints",
                   transform=ax_blank2.transAxes,
                   ha="center", va="center",
                   color="#4cc9f0", fontsize=14, style="italic")

    # Row 3 — Task 5
    add(fig.add_subplot(gs[3, 0:2]), canny_edges,"T5·Canny Edges","Precise boundaries")
    add(fig.add_subplot(gs[3, 2:4]), kp_img,     "T5·ORB Features",
        f"{len(keypoints)} keypoints", True)
    add(fig.add_subplot(gs[3, 4:6]), gray,        "T2·Gray (ref)",  "Original grayscale")

    fig.suptitle(
        "Assignment 5 — Intelligent Image Enhancement: Complete Pipeline",
        fontsize=17, color="white", fontweight="bold", y=0.998)

    save_fig(fig, "master_comparison.png")


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(image_path: str):
    print(f"\n{'━'*65}\n  Processing: {image_path}\n{'━'*65}")

    bgr, gray = acquire_image(image_path)                      # Task 2

    (g_noisy, sp_noisy,                                        # Task 3
     gauss_restored, sp_restored, clahe_img) = enhance_and_restore(gray)

    global_seg, otsu_seg, dilated, eroded = segment_and_morph( # Task 4
        gray, clahe_img)

    canny_edges, contour_img, bbox_img, keypoints = (           # Task 5
        object_and_features(bgr, gray, otsu_seg))

    orb         = cv2.ORB_create(nfeatures=500)
    kps, _      = orb.detectAndCompute(gray, None)
    kp_img      = cv2.drawKeypoints(bgr, kps, None,
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    perf = evaluate_performance(                                 # Task 6
        gray, g_noisy, sp_noisy, gauss_restored, sp_restored, clahe_img)

    final_pipeline_figure(                                       # Task 7
        bgr, gray, g_noisy, sp_noisy,
        gauss_restored, clahe_img,
        otsu_seg, canny_edges, kp_img, perf)

    master_comparison(                                           # Master
        bgr, gray, g_noisy, sp_noisy,
        gauss_restored, sp_restored, clahe_img,
        global_seg, otsu_seg, dilated, eroded,
        canny_edges, kp_img, keypoints)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    print_welcome()

    default_images = [
        "sample_images/image1.jpg",
        "sample_images/image2.jpg",
        "sample_images/image3.jpg",
    ]
    image_paths = sys.argv[1:] if len(sys.argv) > 1 else default_images
    if not sys.argv[1:]:
        print("ℹ  No CLI args — using default_images list.\n"
              "   Run: python main.py img1.jpg img2.jpg img3.jpg\n")

    for idx, path in enumerate(image_paths, 1):
        print(f"\n{'='*65}\n  SAMPLE RUN {idx} / {len(image_paths)}\n{'='*65}")
        try:
            run_pipeline(path)
        except (FileNotFoundError, ValueError) as e:
            print(f"  ⚠  Skipping — {e}")
        except Exception as e:
            import traceback
            print(f"  ❌ Error: {e}")
            traceback.print_exc()

    print(f"\n{'='*65}")
    print("  ✅  All runs complete!")
    print(f"  📁  Outputs: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'='*65}")
    print("\n  Files generated per run:")
    print("    • task2_acquisition.png")
    print("    • task3_enhancement_restoration.png")
    print("    • task4_segmentation_morphology.png")
    print("    • task5_object_features.png")
    print("    • task6_performance_metrics.png")
    print("    • task7_final_pipeline.png")
    print("    • master_comparison.png\n")


if __name__ == "__main__":
    main()
