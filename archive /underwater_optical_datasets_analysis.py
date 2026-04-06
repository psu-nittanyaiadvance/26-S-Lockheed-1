"""
Underwater optical dataset catalog and summary script.

Maintains curated metadata tables for multiple underwater dataset task types
(enhancement, detection, segmentation, tracking, classification) and prints a
summary report for quick comparison.

Data values in this file are static references and can be edited directly as
the catalog evolves.
"""

import pandas as pd

# Enhancement/Restoration Datasets (Weeks 1-5 Focus)
enhancement_datasets = [
    {"year": 2019, "name": "SQUID", "type": "Image+Raw+Depth", "samples": 57, "paired": True, "has_video": False, "has_raw": True, "has_depth": True, "pub": "TPAMI", "description": "RAW images, TIF files, camera calibration files, distance maps from 4 sites", "gs_optical_score": 90},
    {"year": 2019, "name": "U45", "type": "Image+Raw", "samples": 45, "paired": False, "has_video": False, "has_raw": True, "has_depth": False, "pub": "None", "description": "Images with color casts, low contrast and haze-like effects", "gs_optical_score": 40},
    {"year": 2019, "name": "Sea-thru", "type": "Image+Raw+Depth", "samples": None, "paired": True, "has_video": False, "has_raw": True, "has_depth": True, "pub": "CVPR", "description": "RAW images (.ARW/.DEF) with depth maps (.tif)", "gs_optical_score": 95},
    {"year": 2019, "name": "RUIE", "type": "Image+Raw", "samples": None, "paired": False, "has_video": False, "has_raw": True, "has_depth": False, "pub": "TCSVT", "description": "Two subsets: UIQS and UCCS for visibility and color correction", "gs_optical_score": 50},
    {"year": 2019, "name": "UIEB", "type": "Image+Paired", "samples": 950, "paired": True, "has_video": False, "has_raw": False, "has_depth": False, "pub": "TIP", "description": "890 paired images + 60 challenging underwater images", "gs_optical_score": 75},
    {"year": 2020, "name": "EUVP", "type": "Image+Paired+Synthetic", "samples": 20000, "paired": True, "has_video": False, "has_raw": False, "has_depth": False, "pub": "RAL", "description": "12K paired + 8K unpaired instances, CycleGAN-generated pairs", "gs_optical_score": 70},
    {"year": 2020, "name": "OceanDark", "type": "Image+Raw+DeepSea", "samples": 183, "paired": False, "has_video": False, "has_raw": True, "has_depth": False, "pub": "CVPRW", "description": "1280x720 deep-sea images with artificial lighting", "gs_optical_score": 60},
    {"year": 2022, "name": "LNRUD", "type": "Image+Paired+Synthetic", "samples": 50000, "paired": True, "has_video": False, "has_raw": False, "has_depth": False, "pub": "CVPRW", "description": "50K clean + 50K underwater synthesized from 5K real scenes", "gs_optical_score": 65},
    {"year": 2023, "name": "LSUI", "type": "Image+Paired", "samples": 4279, "paired": True, "has_video": False, "has_raw": True, "has_depth": False, "pub": "TIP", "description": "Large-scale dataset with high-quality paired images", "gs_optical_score": 80},
    {"year": 2023, "name": "UVE-38K", "type": "Video+Paired", "samples": 38000, "paired": True, "has_video": True, "has_raw": False, "has_depth": False, "pub": "MTA", "description": "50 video sequences (38K frames) with inter-frame consistent references", "gs_optical_score": 100},  # BEST FOR GAUSSIAN SPLATTING
    {"year": 2023, "name": "DRUVA", "type": "Video+Raw", "samples": None, "paired": False, "has_video": True, "has_raw": True, "has_depth": False, "pub": "ICCV", "description": "20 videos (1 min each) with 360° azimuthal view of artifacts", "gs_optical_score": 95},
    {"year": 2023, "name": "NUID", "type": "Image+Raw+DeepSea", "samples": 925, "paired": False, "has_video": False, "has_raw": True, "has_depth": False, "pub": "TCSVT", "description": "Real underwater images with non-uniform illumination", "gs_optical_score": 55},
    {"year": 2024, "name": "SUVE", "type": "Video+Paired+Synthetic", "samples": 140000, "paired": True, "has_video": True, "has_raw": False, "has_depth": False, "pub": "Arxiv", "description": "660 training + 180 test videos (~170 frames each)", "gs_optical_score": 90},
    {"year": 2024, "name": "UVEB", "type": "Video+Paired", "samples": 453000, "paired": True, "has_video": True, "has_raw": False, "has_depth": False, "pub": "CVPR", "description": "1,308 video sequences, 453K frames (38% UHD 4K)", "gs_optical_score": 100},  # BEST FOR GAUSSIAN SPLATTING
]

# Detection Datasets (for context)
detection_datasets = [
    {"year": 2015, "name": "FishCLEF-2015", "type": "Video", "samples": 14000, "classes": 15, "task": "Detection", "pub": "None"},
    {"year": 2019, "name": "RUIE-UHTS", "type": "Image", "samples": 300, "classes": 3, "task": "Detection", "pub": "TCSVT"},
    {"year": 2019, "name": "Brackish", "type": "Video", "samples": 89, "classes": 6, "task": "Detection", "pub": "CVPRW"},
    {"year": 2019, "name": "Trash-ICRA19", "type": "Image", "samples": 5700, "classes": None, "task": "Detection", "pub": "ICRA"},
    {"year": 2021, "name": "DUO", "type": "Image", "samples": 7782, "classes": 4, "task": "Detection", "pub": "ICME"},
    {"year": 2023, "name": "FishNet", "type": "Image", "samples": 94532, "classes": 17357, "task": "Detection+Classification", "pub": "ICCV"},
]

# Segmentation Datasets
segmentation_datasets = [
    {"year": 2020, "name": "SUIM", "type": "Image", "samples": 1500, "classes": 8, "task": "Segmentation", "pub": "IROS"},
    {"year": 2020, "name": "TrashCan", "type": "Image+Instance", "samples": 7212, "classes": 4, "task": "Segmentation", "pub": "None"},
    {"year": 2023, "name": "UIIS", "type": "Image+Instance", "samples": 4628, "classes": 7, "task": "Segmentation", "pub": "ICCV"},
    {"year": 2024, "name": "USIS10K", "type": "Image+Instance", "samples": 10632, "classes": 7, "task": "Salient Segmentation", "pub": "ICML"},
]

# Tracking Datasets
tracking_datasets = [
    {"year": 2021, "name": "UOT100", "type": "Video", "samples": 74000, "task": "Tracking", "pub": "IJOE"},
    {"year": 2023, "name": "UVOT400", "type": "Video", "samples": 275000, "task": "Tracking", "pub": "Arxiv"},
    {"year": 2024, "name": "WebUOT-1M", "type": "Video", "samples": 1100000, "task": "Tracking+VLT", "pub": "Arxiv"},
]

# Classification Datasets
classification_datasets = [
    {"year": 2012, "name": "Fish4Knowledge", "type": "Image", "samples": 27370, "classes": 23, "task": "Classification", "pub": "ICPR"},
    {"year": 2015, "name": "FishCLEF-2015-Class", "type": "Image", "samples": 20000, "classes": 15, "task": "Classification", "pub": "None"},
    {"year": 2018, "name": "WildFish", "type": "Image", "samples": 54459, "classes": 1000, "task": "Classification", "pub": "ACMMM"},
    {"year": 2023, "name": "FishNet", "type": "Image", "samples": 94532, "classes": 17357, "task": "Classification", "pub": "ICCV"},
]

def print_summary():
    """Print summary statistics"""
    print("="*80)
    print("UNDERWATER OPTICAL DATASETS SUMMARY")
    print("="*80)
    print(f"\nEnhancement/Restoration: {len(enhancement_datasets)} datasets")
    print(f"Object Detection: {len(detection_datasets)} datasets")
    print(f"Semantic Segmentation: {len(segmentation_datasets)} datasets")
    print(f"Object Tracking: {len(tracking_datasets)} datasets")
    print(f"Classification: {len(classification_datasets)} datasets")
    print(f"\nTotal: {len(enhancement_datasets) + len(detection_datasets) + len(segmentation_datasets) + len(tracking_datasets) + len(classification_datasets)} datasets")

    # Top datasets for Gaussian Splatting
    df_enh = pd.DataFrame(enhancement_datasets)
    top_gs = df_enh.nlargest(5, 'gs_optical_score')

    print("\n" + "="*80)
    print("TOP 5 DATASETS FOR GAUSSIAN SPLATTING (Weeks 1-5)")
    print("="*80)
    for idx, row in top_gs.iterrows():
        print(f"\n{row['gs_optical_score']}/100 - {row['name']} ({row['year']})")
        print(f"  Type: {row['type']}")
        samples_str = f"{row['samples']:,}" if pd.notna(row['samples']) else "N/A"
        print(f"  Samples: {samples_str}")
        print(f"  Video: {'✅' if row['has_video'] else '❌'} | Raw: {'✅' if row['has_raw'] else '❌'} | Depth: {'✅' if row['has_depth'] else '❌'}")
        print(f"  {row['description']}")

if __name__ == "__main__":
    print_summary()
