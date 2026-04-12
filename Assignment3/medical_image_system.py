# ============================================================
# Name         : Harsh
# Course       : Image Processing & Computer Vision
# Unit         : Unit 3 — Compression & Segmentation
# Assignment   : Medical Image Compression & Segmentation System
# Date         : 2026
# ============================================================

"""
Medical Image Compression & Segmentation System
------------------------------------------------
Compresses medical images (X-ray / MRI / CT) using Run-Length Encoding
and segments regions of interest via thresholding + morphology.

Per-task image outputs saved to outputs/:
  Task 1 → task1_compression_rle.png
  Task 2 → task2_segmentation_thresholding.png
  Task 3 → task3_morphological_processing.png
  Task 4 → task4_analysis_comparison.png
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

FIG_BG   = "#0a0f1e"
TITLE_KW = dict(fontsize=11, color="white", fontweight="bold", pad=7)
ANNO_KW  = dict(fontsize=8,  color="#8ecae6")


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


# ─────────────────────────────────────────────────────────────
# WELCOME
# ─────────────────────────────────────────────────────────────

def print_welcome():
    print("=" * 65)
    print("   MEDICAL IMAGE COMPRESSION & SEGMENTATION SYSTEM")
    print("=" * 65)
    print("  Course : Image Processing & Computer Vision")
    print("  Unit   : Unit 3 — Compression & Segmentation")
    print("  Author : Harsh")
    print("-" * 65)
    print("  Analyses X-ray / MRI / CT medical images.")
    print("  Compresses using Run-Length Encoding (RLE).")
    print("  Segments ROIs using Global & Otsu's thresholding.")
    print("  Refines boundaries using Dilation & Erosion.")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────
# TASK 1 — IMAGE COMPRESSION (RLE)
# Output: task1_compression_rle.png
# ─────────────────────────────────────────────────────────────

def run_length_encode(row: np.ndarray):
    """
    Lossless Run-Length Encoding for a 1-D row of pixel values.

    Algorithm:
      Traverse the row, counting consecutive identical values.
      Encode as list of (value, count) pairs.

    Returns: list of (pixel_value, run_length) tuples
    """
    if len(row) == 0:
        return []
    encoded = []
    count   = 1
    current = row[0]

    for px in row[1:]:
        if px == current:
            count += 1
        else:
            encoded.append((current, count))
            current = px
            count   = 1
    encoded.append((current, count))
    return encoded


def compute_rle_compression(gray: np.ndarray):
    """
    Apply RLE row-by-row, compute compression ratio and storage savings.
    Returns (original_bits, compressed_bits, ratio, savings_pct).
    """
    original_bits    = gray.size * 8       # 8 bits per pixel
    compressed_bits  = 0

    for row in gray:
        encoded = run_length_encode(row)
        # Each (value, count) pair = 8 bits value + 16 bits count = 24 bits
        compressed_bits += len(encoded) * 24

    ratio        = original_bits / compressed_bits if compressed_bits > 0 else 1.0
    savings_pct  = (1 - compressed_bits / original_bits) * 100

    return original_bits, compressed_bits, ratio, savings_pct


def load_and_compress(image_path: str, target_size=(512, 512)):
    """
    Load medical image → grayscale → RLE compress.
    Saves task1_compression_rle.png with compression stats.
    Returns gray (np.ndarray).
    """
    print(f"\n[Task 1] Image Compression (RLE)")
    print(f"  Source: {image_path}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")

    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    orig_bits, comp_bits, ratio, savings = compute_rle_compression(gray)

    print(f"  Image size        : {target_size[0]}×{target_size[1]} px")
    print(f"  Original size     : {orig_bits:,} bits  ({orig_bits//8:,} bytes)")
    print(f"  Compressed size   : {comp_bits:,} bits  ({comp_bits//8:,} bytes)")
    print(f"  Compression ratio : {ratio:.3f}:1")
    print(f"  Storage savings   : {savings:.1f}%")

    # ── Task 1 figure ─────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Task 1 — Medical Image & RLE Compression Statistics",
                 fontsize=14, color="white", fontweight="bold")

    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1.2])

    # Col 0: original image
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax0.set_title("Grayscale Medical Image", **TITLE_KW)
    ax0.set_xlabel(f"Size: {target_size[0]}×{target_size[1]} px", **ANNO_KW)
    ax0.axis("off")

    # Col 1: pixel histogram (data redundancy visualisation)
    ax1 = fig.add_subplot(gs[1])
    ax1.set_facecolor("#10152a")
    ax1.hist(gray.ravel(), bins=64, color="#4cc9f0", edgecolor="none",
             alpha=0.85)
    ax1.set_title("Pixel Intensity Histogram\n(data redundancy indicator)",
                  **TITLE_KW)
    ax1.set_xlabel("Gray Level", color="white", fontsize=9)
    ax1.set_ylabel("Count",      color="white", fontsize=9)
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#222244")

    # Col 2: compression stats bar chart
    ax2 = fig.add_subplot(gs[2])
    ax2.set_facecolor("#10152a")
    labels = ["Original\n(bits)", "Compressed\n(bits)"]
    values = [orig_bits, comp_bits]
    bars   = ax2.bar(labels, values, color=["#f72585", "#7209b7"],
                     edgecolor="white", linewidth=0.7)
    ax2.set_title(f"RLE Compression\nRatio: {ratio:.2f}:1 | Saved: {savings:.1f}%",
                  **TITLE_KW)
    ax2.set_ylabel("Bits", color="white", fontsize=9)
    ax2.tick_params(colors="white")
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() * 1.02,
                 f"{val//1000:,}K", ha="center",
                 color="white", fontsize=9, fontweight="bold")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#222244")

    plt.tight_layout()
    save_fig(fig, "task1_compression_rle.png")

    return gray


# ─────────────────────────────────────────────────────────────
# TASK 2 — IMAGE SEGMENTATION
# Output: task2_segmentation_thresholding.png
# ─────────────────────────────────────────────────────────────

def segment_image(gray: np.ndarray):
    """
    Apply Global (fixed) and Otsu's adaptive thresholding.
    Saves task2_segmentation_thresholding.png.
    Returns (global_seg, otsu_seg).
    """
    print("\n[Task 2] Image Segmentation — Thresholding")

    # Global threshold at midpoint
    global_thresh = 127
    _, global_seg = cv2.threshold(gray, global_thresh, 255,
                                  cv2.THRESH_BINARY)

    # Otsu's method — automatically finds optimal threshold
    otsu_thresh, otsu_seg = cv2.threshold(gray, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Count foreground pixels (regions of interest)
    fg_global = np.count_nonzero(global_seg)
    fg_otsu   = np.count_nonzero(otsu_seg)

    print(f"  Global threshold   : T = {global_thresh}")
    print(f"  Otsu threshold     : T = {otsu_thresh:.1f} (auto-computed)")
    print(f"  Foreground pixels (Global) : {fg_global:,}  "
          f"({100*fg_global/gray.size:.1f}%)")
    print(f"  Foreground pixels (Otsu)   : {fg_otsu:,}  "
          f"({100*fg_otsu/gray.size:.1f}%)")

    # ── Task 2 figure ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Task 2 — Segmentation: Global vs Otsu's Thresholding",
                 fontsize=14, color="white", fontweight="bold")

    data = [
        ("Original Grayscale",        gray,       "Input image"),
        (f"Global Threshold\nT={global_thresh} (fixed)", global_seg,
         f"Foreground: {100*fg_global/gray.size:.1f}% of pixels"),
        (f"Otsu's Threshold\nT={otsu_thresh:.0f} (auto)", otsu_seg,
         f"Foreground: {100*fg_otsu/gray.size:.1f}% of pixels"),
    ]

    for ax, (title, img, note) in zip(axes, data):
        cmap = "gray" if img.ndim == 2 else None
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, **TITLE_KW)
        ax.set_xlabel(note, **ANNO_KW)
        ax.axis("off")

    plt.tight_layout()
    save_fig(fig, "task2_segmentation_thresholding.png")

    return global_seg, otsu_seg


# ─────────────────────────────────────────────────────────────
# TASK 3 — MORPHOLOGICAL PROCESSING
# Output: task3_morphological_processing.png
# ─────────────────────────────────────────────────────────────

def morphological_processing(gray: np.ndarray,
                              global_seg: np.ndarray,
                              otsu_seg: np.ndarray):
    """
    Apply Dilation and Erosion to both segmented images.
    Saves task3_morphological_processing.png.
    Returns dict of morphologically refined images.
    """
    print("\n[Task 3] Morphological Processing — Dilation & Erosion")

    # Structuring element (kernel) — 5×5 cross
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    results = {}
    for name, seg in [("Global", global_seg), ("Otsu", otsu_seg)]:
        dilated = cv2.dilate(seg, kernel, iterations=2)
        eroded  = cv2.erode(seg,  kernel, iterations=2)

        # Combined: opening = erosion then dilation (removes small noise)
        opened  = cv2.morphologyEx(seg, cv2.MORPH_OPEN,  kernel)
        # Closing = dilation then erosion (fills small holes in ROI)
        closed  = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, kernel)

        results[name] = {
            "Original Seg": seg,
            "Dilation":     dilated,
            "Erosion":      eroded,
            "Opening":      opened,
            "Closing":      closed,
        }

        fg_dil = np.count_nonzero(dilated)
        fg_ero = np.count_nonzero(eroded)
        print(f"  {name} segmentation:")
        print(f"    Dilation  → foreground: {fg_dil:,} px")
        print(f"    Erosion   → foreground: {fg_ero:,} px")

    # ── Task 3 figure: 2 rows (Global, Otsu) × 5 cols ─────────
    fig, axes = plt.subplots(2, 5, figsize=(24, 8))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Task 3 — Morphological Processing: Dilation & Erosion",
                 fontsize=14, color="white", fontweight="bold")

    for r, (seg_name, ops) in enumerate(results.items()):
        for c, (op_name, img) in enumerate(ops.items()):
            axes[r][c].imshow(img, cmap="gray", vmin=0, vmax=255)
            axes[r][c].set_title(
                f"{seg_name} → {op_name}", **TITLE_KW)
            axes[r][c].set_xlabel(
                f"Foreground: {100*np.count_nonzero(img)/img.size:.1f}%",
                **ANNO_KW)
            axes[r][c].axis("off")

    plt.tight_layout()
    save_fig(fig, "task3_morphological_processing.png")

    return results


# ─────────────────────────────────────────────────────────────
# TASK 4 — ANALYSIS & INTERPRETATION
# Output: task4_analysis_comparison.png
# ─────────────────────────────────────────────────────────────

def analysis_and_interpretation(gray: np.ndarray,
                                 global_seg: np.ndarray,
                                 otsu_seg: np.ndarray,
                                 morph_results: dict):
    """
    Compare segmentation results and discuss clinical relevance.
    Saves task4_analysis_comparison.png.
    """
    print("\n" + "=" * 65)
    print("[Task 4] Analysis & Interpretation")
    print("=" * 65)

    print("""
  Segmentation Comparison
  ─────────────────────────────────────────────────────────
  Global Thresholding (T = 127):
    · Uses a fixed midpoint threshold for all images.
    · Fast and simple, but assumes uniform illumination.
    · Fails on medical images with non-uniform backgrounds
      (e.g. MRI with bright skull ring and dark tissue).

  Otsu's Thresholding:
    · Automatically computes T that minimises intra-class variance.
    · Adapts to the actual intensity distribution of each scan.
    · Significantly better on X-rays, MRI, CT where tissue types
      have distinct but varying intensity ranges.
    · Preferred in clinical workflows for ROI detection.

  Morphological Refinement:
    · Dilation: expands ROI boundaries — useful for capturing
      tumour margins or bone edges fully.
    · Erosion: shrinks ROI — removes thin noise connections
      between separate anatomical structures.
    · Opening (Erosion → Dilation): removes small isolated noise
      blobs outside the true ROI.
    · Closing (Dilation → Erosion): fills small gaps / holes
      inside the ROI — useful for solid tumour delineation.

  Clinical Relevance:
    · X-ray chest: segment lung fields → measure opacification.
    · MRI brain:   segment tumour vs healthy tissue.
    · CT abdomen:  isolate organ boundaries for volumetry.
    · Morphological closing is critical for solid masses to
      ensure contiguous segmentation for volume estimation.
""")

    # ── Task 4 figure: side-by-side full comparison ───────────
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Task 4 — Analysis: Segmentation & Morphology Comparison",
                 fontsize=14, color="white", fontweight="bold")

    cols = ["Original\nGrayscale",
            "Global Seg", "Otsu Seg",
            "Otsu → Closing\n(refined)"]

    imgs_row0 = [gray, global_seg, otsu_seg,
                 morph_results["Otsu"]["Closing"]]
    imgs_row1 = [gray,
                 morph_results["Global"]["Dilation"],
                 morph_results["Otsu"]["Dilation"],
                 morph_results["Otsu"]["Erosion"]]
    titles_r1 = ["Original\nGrayscale",
                 "Global → Dilation", "Otsu → Dilation",
                 "Otsu → Erosion"]

    for c, (title, img) in enumerate(zip(cols, imgs_row0)):
        axes[0][c].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[0][c].set_title(title, **TITLE_KW)
        axes[0][c].set_xlabel(
            f"FG: {100*np.count_nonzero(img)/img.size:.1f}%",
            **ANNO_KW)
        axes[0][c].axis("off")

    for c, (title, img) in enumerate(zip(titles_r1, imgs_row1)):
        axes[1][c].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[1][c].set_title(title, **TITLE_KW)
        axes[1][c].set_xlabel(
            f"FG: {100*np.count_nonzero(img)/img.size:.1f}%",
            **ANNO_KW)
        axes[1][c].axis("off")

    plt.tight_layout()
    save_fig(fig, "task4_analysis_comparison.png")


# ─────────────────────────────────────────────────────────────
# MASTER COMPARISON
# Output: master_comparison.png
# ─────────────────────────────────────────────────────────────

def master_comparison(gray, global_seg, otsu_seg, morph_results,
                       comp_ratio, comp_savings):
    """Large summary figure covering all four tasks."""
    fig, axes = plt.subplots(3, 4, figsize=(24, 14))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle(
        "Assignment 3 — Medical Image Compression & Segmentation: Full Pipeline",
        fontsize=16, color="white", fontweight="bold")

    # Row 0: original + RLE stats text + histogram + blank
    axes[0][0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0][0].set_title("Task 1 · Medical Image", **TITLE_KW)
    axes[0][0].axis("off")

    axes[0][1].set_facecolor("#10152a")
    axes[0][1].hist(gray.ravel(), bins=64, color="#4cc9f0",
                    edgecolor="none", alpha=0.85)
    axes[0][1].set_title(
        f"Task 1 · RLE Compression\nRatio: {comp_ratio:.2f}:1 | Saved: {comp_savings:.1f}%",
        **TITLE_KW)
    axes[0][1].tick_params(colors="white")
    for sp in axes[0][1].spines.values():
        sp.set_edgecolor("#222244")

    axes[0][2].imshow(global_seg, cmap="gray", vmin=0, vmax=255)
    axes[0][2].set_title("Task 2 · Global Threshold", **TITLE_KW)
    axes[0][2].axis("off")

    axes[0][3].imshow(otsu_seg, cmap="gray", vmin=0, vmax=255)
    axes[0][3].set_title("Task 2 · Otsu's Threshold", **TITLE_KW)
    axes[0][3].axis("off")

    # Row 1: morphological operations on Global
    ops_global = list(morph_results["Global"].items())
    for c, (op_name, img) in enumerate(ops_global[:4]):
        axes[1][c].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[1][c].set_title(f"Task 3 · Global → {op_name}", **TITLE_KW)
        axes[1][c].axis("off")

    # Row 2: morphological operations on Otsu
    ops_otsu = list(morph_results["Otsu"].items())
    for c, (op_name, img) in enumerate(ops_otsu[:4]):
        axes[2][c].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[2][c].set_title(f"Task 3 · Otsu → {op_name}", **TITLE_KW)
        axes[2][c].axis("off")

    plt.tight_layout()
    save_fig(fig, "master_comparison.png")


# ─────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────

def process_medical_image(image_path: str):
    print(f"\n{'━'*65}\n  Processing: {image_path}\n{'━'*65}")

    gray                   = load_and_compress(image_path)      # Task 1
    global_seg, otsu_seg   = segment_image(gray)                # Task 2
    morph_results          = morphological_processing(          # Task 3
                                gray, global_seg, otsu_seg)
    analysis_and_interpretation(                                 # Task 4
        gray, global_seg, otsu_seg, morph_results)

    # Re-compute for master
    _, _, ratio, savings = compute_rle_compression(gray)
    master_comparison(gray, global_seg, otsu_seg,
                      morph_results, ratio, savings)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    print_welcome()

    default_images = [
        "sample_images/xray.jpg",
        "sample_images/mri.jpg",
        "sample_images/ct_scan.jpg",
    ]
    image_paths = sys.argv[1:] if len(sys.argv) > 1 else default_images
    if not sys.argv[1:]:
        print("ℹ  No CLI args — using default_images list.\n"
              "   Run: python medical_image_system.py xray.jpg mri.jpg ct.jpg\n")

    for idx, path in enumerate(image_paths, 1):
        print(f"\n{'='*65}\n  SAMPLE RUN {idx} / {len(image_paths)}\n{'='*65}")
        try:
            process_medical_image(path)
        except (FileNotFoundError, ValueError) as e:
            print(f"  ⚠  Skipping — {e}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n{'='*65}")
    print("  ✅  All runs complete!")
    print(f"  📁  Outputs: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'='*65}")
    print("\n  Files generated per run:")
    print("    • task1_compression_rle.png")
    print("    • task2_segmentation_thresholding.png")
    print("    • task3_morphological_processing.png")
    print("    • task4_analysis_comparison.png")
    print("    • master_comparison.png\n")


if __name__ == "__main__":
    main()
