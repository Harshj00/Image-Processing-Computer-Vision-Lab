# ============================================================
# Name         : Harsh
# Course       : Image Processing & Computer Vision
# Unit         : Unit 4 — Object Representation & Feature Extraction
# Assignment   : Feature-Based Traffic Monitoring System
# Date         : 2026
# ============================================================

"""
Feature-Based Traffic Monitoring System
----------------------------------------
Detects edges, represents objects via contours/bounding boxes,
and extracts keypoint features from traffic scene images.

Per-task image outputs saved to outputs/:
  Task 1 → task1_edge_detection.png
  Task 2 → task2_object_representation.png
  Task 3 → task3_feature_extraction.png
  Task 4 → task4_comparative_analysis.png
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

FIG_BG   = "#0f0f23"
TITLE_KW = dict(fontsize=11, color="white", fontweight="bold", pad=7)
ANNO_KW  = dict(fontsize=8,  color="#a8dadc")


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


# ─────────────────────────────────────────────────────────────
# WELCOME
# ─────────────────────────────────────────────────────────────

def print_welcome():
    print("=" * 65)
    print("   FEATURE-BASED TRAFFIC MONITORING SYSTEM")
    print("=" * 65)
    print("  Course : Image Processing & Computer Vision")
    print("  Unit   : Unit 4 — Object Representation & Features")
    print("  Author : Harsh")
    print("-" * 65)
    print("  Processes traffic scene images to:")
    print("  · Detect edges (Sobel + Canny)")
    print("  · Represent objects via contours and bounding boxes")
    print("  · Extract keypoint features using ORB")
    print("  All results saved as per-task image outputs.")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────
# IMAGE LOADING
# ─────────────────────────────────────────────────────────────

def load_traffic_image(image_path: str, target_size=(640, 480)):
    """Load a traffic scene image and return (bgr, gray)."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    print(f"  Loaded: {image_path}  →  {target_size[0]}×{target_size[1]} px")
    return resized, gray


# ─────────────────────────────────────────────────────────────
# TASK 1 — EDGE DETECTION
# Output: task1_edge_detection.png
# ─────────────────────────────────────────────────────────────

def detect_edges(bgr: np.ndarray, gray: np.ndarray):
    """
    Apply Sobel operator and Canny edge detector.
    Compare edge quality (density, sharpness).
    Saves task1_edge_detection.png.
    Returns (sobel_edges, canny_edges).
    """
    print("\n[Task 1] Edge Detection")

    # ── Sobel operator ────────────────────────────────────────
    # Computes gradient magnitude in X and Y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)   # horizontal grad
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)   # vertical grad
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_edges = np.uint8(np.clip(sobel_mag / sobel_mag.max() * 255, 0, 255))

    # ── Canny edge detector ───────────────────────────────────
    # Two-threshold hysteresis: strong edges kept, weak edges
    # kept only if connected to strong edges
    canny_edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    sobel_density = np.count_nonzero(sobel_edges > 30) / gray.size * 100
    canny_density = np.count_nonzero(canny_edges) / gray.size * 100

    print(f"  Sobel edge density : {sobel_density:.1f}% of pixels")
    print(f"  Canny edge density : {canny_density:.1f}% of pixels")
    print(f"  Sobel — continuous gradient map (thicker edges)")
    print(f"  Canny — binary thin edges (more precise boundaries)")

    # ── Task 1 figure ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Task 1 — Edge Detection: Sobel vs Canny",
                 fontsize=14, color="white", fontweight="bold")

    axes[0].imshow(bgr_to_rgb(bgr))
    axes[0].set_title("Original Traffic Scene", **TITLE_KW)
    axes[0].set_xlabel("Color input image", **ANNO_KW)
    axes[0].axis("off")

    axes[1].imshow(sobel_edges, cmap="hot")
    axes[1].set_title("Sobel Edges\n(gradient magnitude)", **TITLE_KW)
    axes[1].set_xlabel(f"Edge density: {sobel_density:.1f}%", **ANNO_KW)
    axes[1].axis("off")

    axes[2].imshow(canny_edges, cmap="gray")
    axes[2].set_title("Canny Edges\n(hysteresis thresholding)", **TITLE_KW)
    axes[2].set_xlabel(f"Edge density: {canny_density:.1f}%", **ANNO_KW)
    axes[2].axis("off")

    plt.tight_layout()
    save_fig(fig, "task1_edge_detection.png")

    return sobel_edges, canny_edges


# ─────────────────────────────────────────────────────────────
# TASK 2 — OBJECT REPRESENTATION
# Output: task2_object_representation.png
# ─────────────────────────────────────────────────────────────

def represent_objects(bgr: np.ndarray, gray: np.ndarray,
                       canny_edges: np.ndarray):
    """
    Detect contours from Canny edges, draw bounding boxes,
    compute object area and perimeter.
    Saves task2_object_representation.png.
    Returns (contour_img, bbox_img, areas, perimeters).
    """
    print("\n[Task 2] Object Representation — Contours & Bounding Boxes")

    # Find contours from Canny edge map
    contours, hierarchy = cv2.findContours(
        canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by minimum area to remove noise
    min_area = 500
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    # Draw all contours on copies of original
    contour_img = bgr.copy()
    cv2.drawContours(contour_img, valid_contours, -1,
                     color=(0, 255, 100), thickness=2)

    # Draw bounding boxes
    bbox_img = bgr.copy()
    areas, perimeters = [], []

    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area  = cv2.contourArea(cnt)
        peri  = cv2.arcLength(cnt, closed=True)
        areas.append(area)
        perimeters.append(peri)

        # Draw bounding box
        cv2.rectangle(bbox_img, (x, y), (x+w, y+h),
                      color=(255, 80, 0), thickness=2)
        # Label with area
        cv2.putText(bbox_img, f"A:{int(area)}",
                    (x, max(y-5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 0), 1, cv2.LINE_AA)

    print(f"  Total contours found      : {len(contours)}")
    print(f"  Valid (area≥{min_area} px²)   : {len(valid_contours)}")
    if areas:
        print(f"  Largest object area       : {max(areas):,.0f} px²")
        print(f"  Mean object area          : {np.mean(areas):,.0f} px²")
        print(f"  Mean object perimeter     : {np.mean(perimeters):,.0f} px")

    # ── Task 2 figure ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Task 2 — Object Representation: Contours & Bounding Boxes",
                 fontsize=14, color="white", fontweight="bold")

    axes[0].imshow(bgr_to_rgb(bgr))
    axes[0].set_title("Original", **TITLE_KW)
    axes[0].axis("off")

    axes[1].imshow(bgr_to_rgb(contour_img))
    axes[1].set_title(f"Contours\n({len(valid_contours)} objects detected)",
                      **TITLE_KW)
    axes[1].set_xlabel("cv2.findContours → RETR_EXTERNAL", **ANNO_KW)
    axes[1].axis("off")

    axes[2].imshow(bgr_to_rgb(bbox_img))
    axes[2].set_title("Bounding Boxes\n(with area labels)", **TITLE_KW)
    if areas:
        axes[2].set_xlabel(
            f"Mean area: {np.mean(areas):,.0f} px²  |  "
            f"Mean perimeter: {np.mean(perimeters):,.0f} px",
            **ANNO_KW)
    axes[2].axis("off")

    plt.tight_layout()
    save_fig(fig, "task2_object_representation.png")

    return contour_img, bbox_img, areas, perimeters


# ─────────────────────────────────────────────────────────────
# TASK 3 — FEATURE EXTRACTION (ORB)
# Output: task3_feature_extraction.png
# ─────────────────────────────────────────────────────────────

def extract_features(bgr: np.ndarray, gray: np.ndarray):
    """
    Extract features using ORB (Oriented FAST + Rotated BRIEF).
    ORB is patent-free, fast, and suitable for real-time traffic systems.
    Saves task3_feature_extraction.png.
    Returns (keypoints, descriptors).
    """
    print("\n[Task 3] Feature Extraction — ORB Keypoints & Descriptors")

    # ── ORB detector ──────────────────────────────────────────
    orb = cv2.ORB_create(nfeatures=500,      # max keypoints
                          scaleFactor=1.2,   # image pyramid scale
                          nlevels=8,         # pyramid levels
                          edgeThreshold=31,  # border margin
                          patchSize=31)      # descriptor patch size

    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Draw keypoints with rich information (size, angle)
    kp_img = cv2.drawKeypoints(bgr, keypoints, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Size histogram
    kp_sizes = [kp.size for kp in keypoints]
    kp_responses = [kp.response for kp in keypoints]

    print(f"  ORB keypoints found       : {len(keypoints)}")
    print(f"  Descriptor shape          : "
          f"{descriptors.shape if descriptors is not None else 'None'}")
    if kp_sizes:
        print(f"  Mean keypoint size        : {np.mean(kp_sizes):.2f} px")
        print(f"  Mean response strength    : {np.mean(kp_responses):.4f}")

    # ── Task 3 figure ─────────────────────────────────────────
    fig = plt.figure(figsize=(18, 5))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Task 3 — Feature Extraction: ORB Keypoints & Descriptors",
                 fontsize=14, color="white", fontweight="bold")

    gs   = fig.add_gridspec(1, 3)
    ax0  = fig.add_subplot(gs[0])
    ax1  = fig.add_subplot(gs[1])
    ax2  = fig.add_subplot(gs[2])

    ax0.imshow(bgr_to_rgb(bgr))
    ax0.set_title("Original Scene", **TITLE_KW)
    ax0.axis("off")

    ax1.imshow(bgr_to_rgb(kp_img))
    ax1.set_title(f"ORB Keypoints\n({len(keypoints)} detected)", **TITLE_KW)
    ax1.set_xlabel("Rich keypoints: circles show scale & angle", **ANNO_KW)
    ax1.axis("off")

    # Descriptor heatmap (first 32 descriptors, all 32 bytes)
    if descriptors is not None and len(descriptors) >= 8:
        ax2.set_facecolor("#0a0f1e")
        desc_show = descriptors[:min(32, len(descriptors)), :]
        ax2.imshow(desc_show, cmap="inferno", aspect="auto",
                   interpolation="nearest")
        ax2.set_title(f"Descriptor Matrix\n(first {len(desc_show)} descriptors, "
                      f"32-byte binary)", **TITLE_KW)
        ax2.set_xlabel("Descriptor byte index (0–31)", **ANNO_KW)
        ax2.set_ylabel("Keypoint index", color="white", fontsize=8)
        ax2.tick_params(colors="white")
    else:
        ax2.text(0.5, 0.5, "Not enough\nkeypoints",
                 transform=ax2.transAxes, ha="center",
                 color="white", fontsize=12)
        ax2.axis("off")

    plt.tight_layout()
    save_fig(fig, "task3_feature_extraction.png")

    return keypoints, descriptors


# ─────────────────────────────────────────────────────────────
# TASK 4 — COMPARATIVE ANALYSIS
# Output: task4_comparative_analysis.png
# ─────────────────────────────────────────────────────────────

def comparative_analysis(bgr, gray,
                          sobel_edges, canny_edges,
                          contour_img, bbox_img,
                          keypoints, areas, perimeters):
    """
    Compare edge detectors and feature extractor performance.
    Explain relevance to traffic monitoring.
    Saves task4_comparative_analysis.png.
    """
    print("\n" + "=" * 65)
    print("[Task 4] Comparative Analysis")
    print("=" * 65)

    sobel_density = np.count_nonzero(sobel_edges > 30) / gray.size * 100
    canny_density = np.count_nonzero(canny_edges)       / gray.size * 100

    print(f"""
  Edge Detector Comparison
  ─────────────────────────────────────────────────────────
  Sobel:
    · Gradient-based: computes magnitude at each pixel.
    · Produces thick, continuous gradient map.
    · Less precise edges; sensitive to texture noise.
    · Edge pixel density: {sobel_density:.1f}%
    · Best for: rough boundary detection, lane marking gradients.

  Canny:
    · Multi-stage: Gaussian blur → Sobel → NMS → Hysteresis.
    · Produces thin, binary, precise edges.
    · Superior noise suppression via two-threshold filtering.
    · Edge pixel density: {canny_density:.1f}%
    · Best for: vehicle contour detection, licence plate edges.
    · WINNER for traffic monitoring — precise, noise-robust.

  Object Representation
  ─────────────────────────────────────────────────────────
  Contours detected (≥500 px²): {len(areas)} objects
  Mean object area    : {np.mean(areas) if areas else 0:,.0f} px²
  Mean object perimeter: {np.mean(perimeters) if perimeters else 0:,.0f} px
  · Contours model vehicle/pedestrian boundaries for tracking.
  · Bounding boxes feed directly into vehicle counting systems.
  · Area filter removes noise blobs (road texture, foliage).

  Feature Extraction (ORB)
  ─────────────────────────────────────────────────────────
  Keypoints extracted: {len(keypoints)}
  · ORB is patent-free, real-time capable (~50 FPS).
  · Rotation + scale invariant → robust to camera angle changes.
  · Descriptors enable cross-frame vehicle re-identification.
  · Can track individual vehicles across frames using descriptor
    matching (Hamming distance on binary descriptors).

  Traffic Monitoring Applications
  ─────────────────────────────────────────────────────────
  · Canny → contours → bounding boxes = vehicle detection.
  · ORB descriptors → cross-frame matching = vehicle tracking.
  · Area/perimeter → vehicle size classification (car vs truck).
  · Combined pipeline supports speed estimation, queue length,
    red-light violation detection, and incident reporting.
""")

    # ── Task 4 figure: full pipeline overview ─────────────────
    fig, axes = plt.subplots(2, 3, figsize=(20, 9))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Task 4 — Comparative Analysis: Full Traffic Vision Pipeline",
                 fontsize=14, color="white", fontweight="bold")

    kp_img = cv2.drawKeypoints(bgr, keypoints, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    row0 = [
        ("Original Scene",        bgr,         True),
        ("Sobel Edges",           sobel_edges, False),
        ("Canny Edges",           canny_edges, False),
    ]
    row1 = [
        ("Contours (vehicles)",   contour_img, True),
        ("Bounding Boxes",        bbox_img,    True),
        ("ORB Keypoints",         kp_img,      True),
    ]

    notes0 = [
        "Input traffic scene",
        f"Gradient map — density {sobel_density:.1f}%",
        f"Binary edges — density {canny_density:.1f}%",
    ]
    notes1 = [
        f"{len(areas)} objects detected",
        f"Mean area: {np.mean(areas) if areas else 0:,.0f} px²",
        f"{len(keypoints)} keypoints",
    ]

    for r, (row, notes) in enumerate([(row0, notes0), (row1, notes1)]):
        for c, ((title, img, is_color), note) in enumerate(zip(row, notes)):
            if is_color:
                axes[r][c].imshow(bgr_to_rgb(img))
            else:
                axes[r][c].imshow(img, cmap=("hot" if "Sobel" in title else "gray"))
            axes[r][c].set_title(title, **TITLE_KW)
            axes[r][c].set_xlabel(note, **ANNO_KW)
            axes[r][c].axis("off")

    plt.tight_layout()
    save_fig(fig, "task4_comparative_analysis.png")


# ─────────────────────────────────────────────────────────────
# MASTER COMPARISON
# Output: master_comparison.png
# ─────────────────────────────────────────────────────────────

def master_comparison(bgr, gray,
                       sobel_edges, canny_edges,
                       contour_img, bbox_img,
                       keypoints):
    """Full pipeline summary in one figure."""
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle(
        "Assignment 4 — Traffic Monitoring: Full Pipeline Summary",
        fontsize=16, color="white", fontweight="bold")

    kp_img = cv2.drawKeypoints(bgr, keypoints, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    all_panels = [
        # Row 0
        ("Task 1 · Original",     bgr,         True,  "Traffic scene input"),
        ("Task 1 · Sobel Edges",  sobel_edges, False, "Gradient magnitude map"),
        ("Task 1 · Canny Edges",  canny_edges, False, "Binary precise edges"),
        ("Task 2 · Contours",     contour_img, True,  "Object boundaries"),
        # Row 1
        ("Task 2 · Bounding Boxes", bbox_img,  True,  "Vehicle ROI boxes"),
        ("Task 3 · ORB Features", kp_img,      True,  f"{len(keypoints)} keypoints"),
        ("Grayscale",             gray,         False, "Preprocessing stage"),
        ("Task 4 · Pipeline →",   bgr,         True,  "End-to-end ITS system"),
    ]

    for idx, (title, img, is_color, note) in enumerate(all_panels):
        r, c = divmod(idx, 4)
        if is_color:
            axes[r][c].imshow(bgr_to_rgb(img))
        else:
            axes[r][c].imshow(img, cmap="gray" if "Sobel" not in title else "hot")
        axes[r][c].set_title(title, **TITLE_KW)
        axes[r][c].set_xlabel(note, **ANNO_KW)
        axes[r][c].axis("off")

    plt.tight_layout()
    save_fig(fig, "master_comparison.png")


# ─────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────

def process_traffic_image(image_path: str):
    print(f"\n{'━'*65}\n  Processing: {image_path}\n{'━'*65}")

    bgr, gray           = load_traffic_image(image_path)
    sobel, canny        = detect_edges(bgr, gray)                  # Task 1
    contour_img, bbox_img, areas, peris = represent_objects(       # Task 2
                            bgr, gray, canny)
    keypoints, descs    = extract_features(bgr, gray)              # Task 3
    comparative_analysis(bgr, gray, sobel, canny,                  # Task 4
                         contour_img, bbox_img, keypoints,
                         areas, peris)
    master_comparison(bgr, gray, sobel, canny,                     # Master
                      contour_img, bbox_img, keypoints)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    print_welcome()

    default_images = [
        "sample_images/road_intersection.jpg",
        "sample_images/highway.jpg",
        "sample_images/pedestrian_crossing.jpg",
    ]
    image_paths = sys.argv[1:] if len(sys.argv) > 1 else default_images
    if not sys.argv[1:]:
        print("ℹ  No CLI args — using default_images list.\n"
              "   Run: python traffic_monitoring.py img1.jpg img2.jpg img3.jpg\n")

    for idx, path in enumerate(image_paths, 1):
        print(f"\n{'='*65}\n  SAMPLE RUN {idx} / {len(image_paths)}\n{'='*65}")
        try:
            process_traffic_image(path)
        except (FileNotFoundError, ValueError) as e:
            print(f"  ⚠  Skipping — {e}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n{'='*65}")
    print("  ✅  All runs complete!")
    print(f"  📁  Outputs: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'='*65}")
    print("\n  Files generated per run:")
    print("    • task1_edge_detection.png")
    print("    • task2_object_representation.png")
    print("    • task3_feature_extraction.png")
    print("    • task4_comparative_analysis.png")
    print("    • master_comparison.png\n")


if __name__ == "__main__":
    main()
