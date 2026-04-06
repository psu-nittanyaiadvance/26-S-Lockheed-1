#!/usr/bin/env python3
"""
Rank underwater images by relevance and quality.

This script scores images using a hybrid of semantic signals (CLIP prompt
similarity) and lightweight visual heuristics, then retains a top-K subset.

Two-pass approach:
    Pass 1: Stream images, compute scores, maintain a top-K min-heap.
    Pass 2: Stream again to attach neighbor context around selected items.
"""

import argparse
import csv
import heapq
import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import (
    Iterator, List, Optional, Tuple, Dict, Any, Set, Generator
)
import warnings

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from PIL import Image, ImageFile
import torch
from tqdm import tqdm

# Allow truncated images to be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Optional OpenCV import with graceful fallback
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logging.warning("OpenCV not available; edge density scoring disabled.")

# Transformers import
from transformers import CLIPProcessor, CLIPModel

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_K = 200
DEFAULT_BATCH_SIZE = 16
DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
WINDOW_SIZE = 11  # 5 prev + 1 current + 5 next

# Positive prompts: describe useful sonar/underwater/perception images
POSITIVE_PROMPTS = [
    "a clear sonar image showing seabed features and underwater structures",
    "a high quality side-scan sonar image with good contrast",
    "an underwater camera image with visible objects and clear visibility",
    "a detailed acoustic image showing seafloor texture and marine objects",
    "a surround camera image from underwater vehicle with good lighting",
    "a sonar scan showing distinct edges and object boundaries",
    "bathymetric data visualization with clear depth contours",
    "underwater terrain mapping image with identifiable features",
]

# Negative prompts: describe bad/useless images to penalize
NEGATIVE_PROMPTS = [
    "a completely black empty image with no content",
    "a pure white overexposed blank image",
    "random noise static corruption artifacts",
    "an extremely blurry out of focus image",
    "a heavily compressed jpeg with blocking artifacts",
    "corrupted unreadable garbled image data",
    "a blank featureless uniform gray image",
]

# Default scoring weights
DEFAULT_WEIGHTS = {
    "clip_positive": 0.35,      # CLIP similarity to positive prompts
    "clip_negative": 0.15,      # CLIP dissimilarity to negative prompts (inverted)
    "entropy": 0.15,            # Grayscale entropy
    "laplacian_var": 0.15,      # Laplacian variance (sharpness)
    "saturation_penalty": 0.10, # Penalty for saturated pixels
    "edge_density": 0.10,       # Edge density (Canny)
}


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class ImageScore:
    """Score record for a single image."""
    index: int
    path: str
    score: float
    components: Dict[str, float] = field(default_factory=dict)


@dataclass
class RankedImageRecord:
    """Full record for a top-K image with neighbor context."""
    rank: int
    path: str
    relative_path: str
    score: float
    score_components: Dict[str, float]
    prev_5_paths: List[str]
    next_5_paths: List[str]


# -----------------------------------------------------------------------------
# Image Path Generator (Memory-Safe)
# -----------------------------------------------------------------------------

def iter_image_paths(
    root: Path,
    extensions: Set[str]
) -> Generator[Tuple[int, str, str], None, None]:
    """
    Yield (index, absolute_path, relative_path) for all images under root,
    sorted deterministically by relative path.

    Uses os.walk with sorted() to ensure deterministic order without loading
    all paths into memory at once for the initial scan.

    Note: We do need to collect paths for sorting. For truly massive datasets
    (billions of files), you'd want external sorting. For millions, this is fine.
    """
    # Collect relative paths first (necessary for global sorted order)
    # This is O(n) memory but unavoidable for deterministic global ordering
    all_paths: List[Tuple[str, str]] = []

    root_str = str(root.resolve())
    for dirpath, _, filenames in os.walk(root_str):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in extensions:
                abs_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(abs_path, root_str)
                all_paths.append((rel_path, abs_path))

    # Sort by relative path for deterministic order
    all_paths.sort(key=lambda x: x[0])

    # Yield with index
    for idx, (rel_path, abs_path) in enumerate(all_paths):
        yield idx, abs_path, rel_path


def count_images(root: Path, extensions: Set[str]) -> int:
    """Count total images for progress bar."""
    count = 0
    for _, _, filenames in os.walk(str(root)):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in extensions:
                count += 1
    return count


# -----------------------------------------------------------------------------
# CV Heuristics (Fast, CPU-based)
# -----------------------------------------------------------------------------

def compute_grayscale_entropy(img_array: np.ndarray) -> float:
    """
    Compute Shannon entropy of grayscale histogram.
    Higher entropy = more information content.
    Returns value in [0, 8] for 8-bit images, normalized to [0, 1].
    """
    if img_array.ndim == 3:
        # Convert to grayscale using luminosity method
        gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        gray = img_array

    # Compute histogram
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    hist = hist[hist > 0]  # Remove zero bins

    # Normalize to probability distribution
    hist = hist / hist.sum()

    # Shannon entropy
    entropy = -np.sum(hist * np.log2(hist))

    # Normalize to [0, 1] (max entropy for 8-bit is 8)
    return min(entropy / 8.0, 1.0)


def compute_laplacian_variance(img_array: np.ndarray) -> float:
    """
    Compute Laplacian variance as a measure of image sharpness.
    Higher variance = sharper image.
    Uses a simple 3x3 Laplacian kernel.
    """
    if img_array.ndim == 3:
        gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
    else:
        gray = img_array.astype(np.float32)

    # Simple Laplacian kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    # Manual convolution using numpy (avoid OpenCV dependency here)
    from scipy.ndimage import convolve
    laplacian = convolve(gray, kernel, mode='reflect')

    variance = laplacian.var()

    # Normalize: typical range is 0-5000+, use sigmoid-like mapping
    normalized = 2.0 / (1.0 + np.exp(-variance / 500.0)) - 1.0
    return float(np.clip(normalized, 0, 1))


def compute_saturation_penalty(img_array: np.ndarray) -> float:
    """
    Compute penalty for saturated (pure black/white) pixels.
    Returns 1.0 if no saturation, approaches 0.0 with heavy saturation.
    """
    if img_array.ndim == 3:
        gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img_array.astype(np.float64)

    total_pixels = gray.size

    # Count near-black and near-white pixels
    near_black = np.sum(gray < 5)
    near_white = np.sum(gray > 250)

    saturated_ratio = (near_black + near_white) / total_pixels

    # Penalty: 1.0 means no saturation, 0.0 means fully saturated
    # Use exponential decay
    penalty = np.exp(-5.0 * saturated_ratio)
    return float(penalty)


def compute_edge_density(img_array: np.ndarray) -> float:
    """
    Compute edge density using Canny edge detection (requires OpenCV).
    Returns ratio of edge pixels to total pixels, normalized.
    Falls back to gradient-based approximation if OpenCV unavailable.
    """
    if img_array.ndim == 3:
        gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        gray = img_array.astype(np.uint8)

    if HAS_OPENCV:
        # Use Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
    else:
        # Fallback: simple gradient magnitude
        gx = np.diff(gray.astype(np.float32), axis=1)
        gy = np.diff(gray.astype(np.float32), axis=0)
        # Crop to same size
        gx = gx[:-1, :]
        gy = gy[:, :-1]
        magnitude = np.sqrt(gx**2 + gy**2)
        edge_ratio = np.sum(magnitude > 30) / magnitude.size

    # Normalize: typical good images have 1-10% edges
    normalized = min(edge_ratio * 10, 1.0)
    return float(normalized)


def compute_cv_heuristics(img_array: np.ndarray) -> Dict[str, float]:
    """Compute all CV heuristic scores for an image array."""
    try:
        return {
            "entropy": compute_grayscale_entropy(img_array),
            "laplacian_var": compute_laplacian_variance(img_array),
            "saturation_penalty": compute_saturation_penalty(img_array),
            "edge_density": compute_edge_density(img_array),
        }
    except Exception as e:
        logging.warning(f"CV heuristics failed: {e}")
        return {
            "entropy": 0.0,
            "laplacian_var": 0.0,
            "saturation_penalty": 0.0,
            "edge_density": 0.0,
        }


# -----------------------------------------------------------------------------
# CLIP-based Scoring
# -----------------------------------------------------------------------------

class CLIPScorer:
    """
    Handles CLIP model loading and embedding computation.
    Precomputes text embeddings for positive/negative prompts.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        positive_prompts: List[str] = None,
        negative_prompts: List[str] = None,
    ):
        self.device = device
        self.model_name = model_name

        logging.info(f"Loading CLIP model: {model_name} on {device}")

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        # Precompute text embeddings
        pos_prompts = positive_prompts or POSITIVE_PROMPTS
        neg_prompts = negative_prompts or NEGATIVE_PROMPTS

        with torch.no_grad():
            # Positive prompts
            pos_inputs = self.processor(
                text=pos_prompts, return_tensors="pt", padding=True, truncation=True
            )
            pos_inputs = {k: v.to(device) for k, v in pos_inputs.items() if k != "pixel_values"}
            self.pos_text_embeds = self.model.get_text_features(**pos_inputs)
            self.pos_text_embeds = self.pos_text_embeds / self.pos_text_embeds.norm(dim=-1, keepdim=True)

            # Negative prompts
            neg_inputs = self.processor(
                text=neg_prompts, return_tensors="pt", padding=True, truncation=True
            )
            neg_inputs = {k: v.to(device) for k, v in neg_inputs.items() if k != "pixel_values"}
            self.neg_text_embeds = self.model.get_text_features(**neg_inputs)
            self.neg_text_embeds = self.neg_text_embeds / self.neg_text_embeds.norm(dim=-1, keepdim=True)

        logging.info(f"Precomputed embeddings for {len(pos_prompts)} positive, {len(neg_prompts)} negative prompts")

    def compute_image_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """Compute normalized image embeddings for a batch of PIL images."""
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        return image_embeds

    def score_batch(self, images: List[Image.Image]) -> List[Dict[str, float]]:
        """
        Compute CLIP scores for a batch of images.
        Returns list of dicts with 'clip_positive' and 'clip_negative' scores.
        """
        if not images:
            return []

        image_embeds = self.compute_image_embeddings(images)

        # Cosine similarity with positive prompts (mean across prompts)
        pos_sim = torch.matmul(image_embeds, self.pos_text_embeds.T)  # [batch, n_pos]
        pos_scores = pos_sim.mean(dim=-1)  # [batch]

        # Cosine similarity with negative prompts (mean across prompts)
        neg_sim = torch.matmul(image_embeds, self.neg_text_embeds.T)  # [batch, n_neg]
        neg_scores = neg_sim.mean(dim=-1)  # [batch]

        # Convert to numpy and normalize to [0, 1]
        # CLIP similarities are typically in [-1, 1], shift to [0, 1]
        pos_scores = ((pos_scores.cpu().numpy() + 1) / 2).clip(0, 1)
        neg_scores = ((neg_scores.cpu().numpy() + 1) / 2).clip(0, 1)

        # For negative, we want LOW similarity to be GOOD, so invert
        neg_scores_inverted = 1.0 - neg_scores

        return [
            {"clip_positive": float(p), "clip_negative": float(n)}
            for p, n in zip(pos_scores, neg_scores_inverted)
        ]


# -----------------------------------------------------------------------------
# Image Loading Utilities
# -----------------------------------------------------------------------------

def load_image_safe(path: str, max_size: int = 1024) -> Optional[Tuple[Image.Image, np.ndarray]]:
    """
    Safely load an image, handling errors and limiting size.
    Returns (PIL.Image, numpy_array) or None if failed.
    """
    try:
        img = Image.open(path)

        # Convert to RGB (handles grayscale, RGBA, palette, etc.)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize if too large (for memory and speed)
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        arr = np.array(img)

        # Validate array
        if arr.size == 0 or arr.ndim < 2:
            return None

        return img, arr

    except Exception as e:
        logging.debug(f"Failed to load {path}: {e}")
        return None


# -----------------------------------------------------------------------------
# Scoring Pipeline
# -----------------------------------------------------------------------------

def compute_final_score(
    clip_scores: Dict[str, float],
    cv_scores: Dict[str, float],
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """
    Combine CLIP and CV scores into a final score in [0, 1].
    Returns (final_score, component_dict).
    """
    components = {**clip_scores, **cv_scores}

    # Weighted sum
    total_weight = sum(weights.values())
    final_score = 0.0

    for key, weight in weights.items():
        if key in components:
            final_score += weight * components[key]

    # Normalize by total weight
    if total_weight > 0:
        final_score /= total_weight

    return float(np.clip(final_score, 0, 1)), components


# -----------------------------------------------------------------------------
# Two-Pass Ranking Algorithm
# -----------------------------------------------------------------------------

class ImageRanker:
    """
    Main ranking engine using two-pass approach.

    Pass 1: Stream all images, compute scores, maintain top-K min-heap.
    Pass 2: Stream again with rolling window to attach prev/next neighbors.
    """

    def __init__(
        self,
        root: Path,
        k: int = DEFAULT_K,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        extensions: Set[str] = None,
        weights: Dict[str, float] = None,
        seed: int = 42,
    ):
        self.root = root
        self.k = k
        self.batch_size = batch_size
        self.extensions = extensions or DEFAULT_EXTENSIONS
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.seed = seed

        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize CLIP scorer
        self.clip_scorer = CLIPScorer(model_name=model_name, device=device)

        # State for tracking
        self.total_images = 0
        self.processed_images = 0
        self.failed_images = 0
        self.top_k_heap: List[Tuple[float, int, str, str, Dict]] = []  # min-heap

        # Map from index to score record for final top-K
        self.top_k_map: Dict[int, ImageScore] = {}

    def pass1_score_and_rank(self) -> None:
        """
        First pass: stream all images, compute scores, maintain top-K heap.
        Uses min-heap so we can efficiently remove the smallest when at capacity.
        """
        logging.info("Pass 1: Scoring all images...")

        # Count total for progress bar
        self.total_images = count_images(self.root, self.extensions)
        logging.info(f"Found {self.total_images} images to process")

        # Batch accumulator
        batch_data: List[Tuple[int, str, str, Image.Image, np.ndarray]] = []

        pbar = tqdm(
            total=self.total_images,
            desc="Pass 1: Scoring",
            unit="img",
            dynamic_ncols=True,
        )

        for idx, abs_path, rel_path in iter_image_paths(self.root, self.extensions):
            # Try to load image
            result = load_image_safe(abs_path)

            if result is None:
                self.failed_images += 1
                logging.debug(f"Skipped unreadable: {rel_path}")
                pbar.update(1)
                continue

            pil_img, np_arr = result
            batch_data.append((idx, abs_path, rel_path, pil_img, np_arr))

            # Process batch when full
            if len(batch_data) >= self.batch_size:
                self._process_batch(batch_data)
                batch_data = []

            pbar.update(1)

        # Process remaining batch
        if batch_data:
            self._process_batch(batch_data)

        pbar.close()

        self.processed_images = self.total_images - self.failed_images
        logging.info(
            f"Pass 1 complete: {self.processed_images} processed, "
            f"{self.failed_images} failed, {len(self.top_k_heap)} in top-K"
        )

        # Build top-K map for pass 2
        self.top_k_map = {}
        for score, idx, abs_path, rel_path, components in self.top_k_heap:
            self.top_k_map[idx] = ImageScore(
                index=idx, path=abs_path, score=score, components=components
            )

    def _process_batch(
        self,
        batch_data: List[Tuple[int, str, str, Image.Image, np.ndarray]]
    ) -> None:
        """Process a batch of images: compute CLIP + CV scores, update heap."""

        # Extract PIL images for CLIP
        pil_images = [item[3] for item in batch_data]

        # Compute CLIP scores (batched)
        try:
            clip_scores_list = self.clip_scorer.score_batch(pil_images)
        except Exception as e:
            logging.warning(f"CLIP batch scoring failed: {e}")
            clip_scores_list = [{"clip_positive": 0.0, "clip_negative": 0.5}] * len(batch_data)

        # Process each image
        for i, (idx, abs_path, rel_path, pil_img, np_arr) in enumerate(batch_data):
            # Compute CV heuristics
            cv_scores = compute_cv_heuristics(np_arr)

            # Combine scores
            clip_scores = clip_scores_list[i] if i < len(clip_scores_list) else {
                "clip_positive": 0.0, "clip_negative": 0.5
            }
            final_score, components = compute_final_score(
                clip_scores, cv_scores, self.weights
            )

            # Update top-K heap (min-heap)
            entry = (final_score, idx, abs_path, rel_path, components)

            if len(self.top_k_heap) < self.k:
                heapq.heappush(self.top_k_heap, entry)
            elif final_score > self.top_k_heap[0][0]:
                # New score is better than worst in heap
                heapq.heapreplace(self.top_k_heap, entry)

    def pass2_attach_neighbors(self) -> List[RankedImageRecord]:
        """
        Second pass: stream with rolling window to attach prev/next neighbors
        for images in the final top-K set.

        Uses a deque of size 11 (5 prev + current + 5 next) to track context.
        When we see an image that's 5 positions past a top-K candidate,
        we can finalize that candidate's neighbors.
        """
        logging.info("Pass 2: Attaching neighbor context...")

        top_k_indices = set(self.top_k_map.keys())

        # Window: stores (index, rel_path) tuples
        window: deque = deque(maxlen=WINDOW_SIZE)

        # Pending: top-K images waiting for their next-5 neighbors
        # Maps index -> position in window when first seen
        pending: Dict[int, int] = {}

        # Final results
        results: Dict[int, RankedImageRecord] = {}

        # Track window position for each pending item
        pending_window_positions: Dict[int, List[str]] = {}

        pbar = tqdm(
            total=self.total_images,
            desc="Pass 2: Context",
            unit="img",
            dynamic_ncols=True,
        )

        def finalize_pending(target_idx: int, window_list: List[Tuple[int, str]]):
            """Finalize a pending top-K image with its neighbors."""
            score_record = self.top_k_map[target_idx]

            # Find position of target in window
            try:
                target_pos = next(
                    i for i, (idx, _) in enumerate(window_list) if idx == target_idx
                )
            except StopIteration:
                return

            # Extract prev 5 and next 5
            prev_5 = [
                path for idx, path in window_list[max(0, target_pos - 5):target_pos]
            ]
            next_5 = [
                path for idx, path in window_list[target_pos + 1:target_pos + 6]
            ]

            # Get relative path
            rel_path = os.path.relpath(score_record.path, str(self.root))

            results[target_idx] = RankedImageRecord(
                rank=-1,  # Will be set after sorting
                path=score_record.path,
                relative_path=rel_path,
                score=score_record.score,
                score_components=score_record.components,
                prev_5_paths=prev_5,
                next_5_paths=next_5,
            )

        for idx, abs_path, rel_path in iter_image_paths(self.root, self.extensions):
            window.append((idx, rel_path))

            # Check if this index is in top-K
            if idx in top_k_indices and idx not in pending and idx not in results:
                pending[idx] = len(window) - 1

            # Check if any pending items can be finalized
            # (they need 5 more images after them)
            to_finalize = []
            for pending_idx in list(pending.keys()):
                # Find position of pending item in current window
                window_list = list(window)
                try:
                    pos = next(
                        i for i, (widx, _) in enumerate(window_list) if widx == pending_idx
                    )
                    # If there are 5+ images after it, or we're at the end, finalize
                    if len(window_list) - pos > 5:
                        to_finalize.append((pending_idx, window_list))
                except StopIteration:
                    # Item scrolled out of window, finalize with what we have
                    pass

            for pending_idx, window_list in to_finalize:
                finalize_pending(pending_idx, window_list)
                del pending[pending_idx]

            pbar.update(1)

        # Finalize any remaining pending items at the end of iteration
        window_list = list(window)
        for pending_idx in pending:
            finalize_pending(pending_idx, window_list)

        pbar.close()

        # Sort by score descending and assign ranks
        sorted_results = sorted(
            results.values(),
            key=lambda x: x.score,
            reverse=True
        )

        for rank, record in enumerate(sorted_results, start=1):
            record.rank = rank

        logging.info(f"Pass 2 complete: {len(sorted_results)} records with context")

        return sorted_results

    def run(self) -> List[RankedImageRecord]:
        """Execute the full two-pass ranking pipeline."""
        self.pass1_score_and_rank()
        return self.pass2_attach_neighbors()


# -----------------------------------------------------------------------------
# Output Writers
# -----------------------------------------------------------------------------

def write_jsonl(records: List[RankedImageRecord], path: Path) -> None:
    """Write records to JSONL format (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            json_obj = {
                "rank": record.rank,
                "path": record.path,
                "relative_path": record.relative_path,
                "score": record.score,
                "score_components": record.score_components,
                "prev_5_paths": record.prev_5_paths,
                "next_5_paths": record.next_5_paths,
            }
            f.write(json.dumps(json_obj) + "\n")
    logging.info(f"Wrote JSONL: {path}")


def write_summary_json(
    records: List[RankedImageRecord],
    metadata: Dict[str, Any],
    path: Path
) -> None:
    """Write full summary JSON with metadata and ranked list."""
    summary = {
        "metadata": metadata,
        "ranked_images": [
            {
                "rank": r.rank,
                "path": r.path,
                "relative_path": r.relative_path,
                "score": r.score,
                "score_components": r.score_components,
                "prev_5_paths": r.prev_5_paths,
                "next_5_paths": r.next_5_paths,
            }
            for r in records
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Wrote summary JSON: {path}")


def write_csv(records: List[RankedImageRecord], path: Path) -> None:
    """Write records to CSV format."""
    fieldnames = [
        "rank", "relative_path", "score",
        "clip_positive", "clip_negative", "entropy",
        "laplacian_var", "saturation_penalty", "edge_density",
        "prev_5_paths", "next_5_paths"
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            row = {
                "rank": record.rank,
                "relative_path": record.relative_path,
                "score": f"{record.score:.6f}",
                "clip_positive": f"{record.score_components.get('clip_positive', 0):.6f}",
                "clip_negative": f"{record.score_components.get('clip_negative', 0):.6f}",
                "entropy": f"{record.score_components.get('entropy', 0):.6f}",
                "laplacian_var": f"{record.score_components.get('laplacian_var', 0):.6f}",
                "saturation_penalty": f"{record.score_components.get('saturation_penalty', 0):.6f}",
                "edge_density": f"{record.score_components.get('edge_density', 0):.6f}",
                "prev_5_paths": ";".join(record.prev_5_paths),
                "next_5_paths": ";".join(record.next_5_paths),
            }
            writer.writerow(row)
    logging.info(f"Wrote CSV: {path}")


# -----------------------------------------------------------------------------
# CLI Argument Parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank images by usefulness for underwater autonomy (sonar + surround camera).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required paths
    parser.add_argument(
        "--root", "-r",
        type=str,
        required=True,
        help="Root directory containing images (will be walked recursively).",
    )
    parser.add_argument(
        "--outdir", "-o",
        type=str,
        default="ranked_output",
        help="Output directory for results.",
    )

    # Top-K configuration
    parser.add_argument(
        "--k", "-k",
        type=int,
        default=DEFAULT_K,
        help="Number of top images to keep.",
    )

    # Model configuration
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face CLIP model identifier.",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device for inference. 'auto' uses CUDA if available.",
    )

    # Processing configuration
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for CLIP inference.",
    )
    parser.add_argument(
        "--extensions", "-e",
        type=str,
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help="Image file extensions to include.",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    # Scoring weights
    parser.add_argument(
        "--weight-clip-positive",
        type=float,
        default=DEFAULT_WEIGHTS["clip_positive"],
        help="Weight for CLIP positive prompt similarity.",
    )
    parser.add_argument(
        "--weight-clip-negative",
        type=float,
        default=DEFAULT_WEIGHTS["clip_negative"],
        help="Weight for CLIP negative prompt dissimilarity.",
    )
    parser.add_argument(
        "--weight-entropy",
        type=float,
        default=DEFAULT_WEIGHTS["entropy"],
        help="Weight for grayscale entropy score.",
    )
    parser.add_argument(
        "--weight-laplacian",
        type=float,
        default=DEFAULT_WEIGHTS["laplacian_var"],
        help="Weight for Laplacian variance (sharpness) score.",
    )
    parser.add_argument(
        "--weight-saturation",
        type=float,
        default=DEFAULT_WEIGHTS["saturation_penalty"],
        help="Weight for saturation penalty score.",
    )
    parser.add_argument(
        "--weight-edge",
        type=float,
        default=DEFAULT_WEIGHTS["edge_density"],
        help="Weight for edge density score.",
    )

    # Output options
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also output results as CSV.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Resolve paths
    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()

    if not root.exists():
        logging.error(f"Root directory does not exist: {root}")
        sys.exit(1)

    outdir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    logging.info(f"Using device: {device}")

    # Build weights dict
    weights = {
        "clip_positive": args.weight_clip_positive,
        "clip_negative": args.weight_clip_negative,
        "entropy": args.weight_entropy,
        "laplacian_var": args.weight_laplacian,
        "saturation_penalty": args.weight_saturation,
        "edge_density": args.weight_edge,
    }

    # Normalize extensions
    extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.extensions}

    logging.info(f"Configuration:")
    logging.info(f"  Root: {root}")
    logging.info(f"  Output: {outdir}")
    logging.info(f"  K: {args.k}")
    logging.info(f"  Model: {args.model}")
    logging.info(f"  Batch size: {args.batch_size}")
    logging.info(f"  Extensions: {extensions}")
    logging.info(f"  Weights: {weights}")

    # Run ranking
    ranker = ImageRanker(
        root=root,
        k=args.k,
        batch_size=args.batch_size,
        model_name=args.model,
        device=device,
        extensions=extensions,
        weights=weights,
        seed=args.seed,
    )

    results = ranker.run()

    # Prepare metadata
    metadata = {
        "root": str(root),
        "k": args.k,
        "model": args.model,
        "device": device,
        "batch_size": args.batch_size,
        "extensions": list(extensions),
        "weights": weights,
        "seed": args.seed,
        "total_images": ranker.total_images,
        "processed_images": ranker.processed_images,
        "failed_images": ranker.failed_images,
        "final_top_k_count": len(results),
    }

    # Write outputs
    write_jsonl(results, outdir / "top_images.jsonl")
    write_summary_json(results, metadata, outdir / "summary.json")

    if args.csv:
        write_csv(results, outdir / "top_images.csv")

    # Print summary
    logging.info("=" * 60)
    logging.info("RANKING COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Total images found: {ranker.total_images}")
    logging.info(f"Successfully processed: {ranker.processed_images}")
    logging.info(f"Failed/skipped: {ranker.failed_images}")
    logging.info(f"Top-K images: {len(results)}")
    logging.info(f"Output directory: {outdir}")

    if results:
        logging.info("\nTop 5 images:")
        for record in results[:5]:
            logging.info(f"  #{record.rank}: {record.relative_path} (score: {record.score:.4f})")


if __name__ == "__main__":
    main()


# =============================================================================
# USAGE EXAMPLES
# =============================================================================
#
# Basic usage (uses defaults: K=200, CPU if no CUDA):
#   python rank_images.py --root raw/images --outdir ranked
#
# With CUDA on RTX 3080:
#   python rank_images.py --root raw/images --outdir ranked --device cuda --batch-size 32
#
# Custom model and larger K:
#   python rank_images.py --root raw/images --model openai/clip-vit-large-patch14 --k 500
#
# Adjust scoring weights (more emphasis on CLIP, less on CV):
#   python rank_images.py --root raw/images --weight-clip-positive 0.5 --weight-entropy 0.1
#
# Include CSV output and verbose logging:
#   python rank_images.py --root raw/images --outdir ranked --csv --verbose
#
# Custom extensions:
#   python rank_images.py --root raw/images --extensions .jpg .png .tiff
#
# Full example with all options:
#   python rank_images.py \
#     --root raw/images \
#     --outdir ranked \
#     --k 200 \
#     --model openai/clip-vit-base-patch32 \
#     --device cuda \
#     --batch-size 16 \
#     --seed 42 \
#     --weight-clip-positive 0.35 \
#     --weight-clip-negative 0.15 \
#     --weight-entropy 0.15 \
#     --weight-laplacian 0.15 \
#     --weight-saturation 0.10 \
#     --weight-edge 0.10 \
#     --csv \
#     --verbose
#
# =============================================================================
