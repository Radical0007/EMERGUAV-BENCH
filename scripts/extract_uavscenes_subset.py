import argparse
import shutil
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emerguav_bench.utils.progress import progress_iter


def copy_file_safe(src: Path, dest: Path) -> bool:
    if src.exists() and src.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Extract a subset of frames and labels from a UAVScenes dataset directory.")
    parser.add_argument("--input-dir", required=True, help="Path to the root of the raw UAVScenes dataset.")
    parser.add_argument("--output-dir", required=True, help="Path where the extracted subset will be saved.")
    parser.add_argument("--frames-per-sequence", type=int, default=50, help="Number of frames to extract per sequence (default: 50).")
    
    args = parser.parse_args()

    input_root = Path(args.input_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()

    if not input_root.exists() or not input_root.is_dir():
        print(f"Error: Input directory '{input_root}' does not exist or is not a directory.")
        return

    print(f"Starting extraction from: {input_root}")
    print(f"Destination: {output_root}")
    print(f"Frames per sequence: {args.frames_per_sequence}")
    print("-" * 50)

    sequence_dirs = []
    for item in progress_iter(list(input_root.iterdir()), desc="scan root", unit="entry"):
        if item.is_dir() and item.name.startswith("interval1_") and not item.name.endswith("_label"):
            sequence_dirs.append(item)

    if not sequence_dirs:
        print("No valid sequence directories found. Ensure the input directory contains folders like 'interval1_AMtown03'.")
        return

    total_sequences = len(sequence_dirs)
    
    stats = {
        "sequences_processed": 0,
        "images_copied": 0,
        "cam_labels_copied": 0,
        "lidars_copied": 0,
        "lidar_labels_copied": 0
    }

    sorted_sequence_dirs = sorted(sequence_dirs)
    for idx, seq_dir in enumerate(progress_iter(sorted_sequence_dirs, desc="extract sequences", total=len(sorted_sequence_dirs), unit="sequence"), 1):
        seq_name = seq_dir.name
        print(f"[{idx}/{total_sequences}] Processing sequence: {seq_name}")
        
        src_cam_dir = seq_dir / "interval1_CAM"
        src_lidar_dir = seq_dir / "interval1_LIDAR"
        
        src_cam_label_dir = input_root / "interval1_CAM_label" / seq_name
        src_lidar_label_dir = input_root / "interval1_LIDAR_label" / seq_name

        dest_seq_dir = output_root / seq_name
        dest_cam_dir = dest_seq_dir / "interval1_CAM"
        dest_lidar_dir = dest_seq_dir / "interval1_LIDAR"
        dest_cam_label_dir = output_root / "interval1_CAM_label" / seq_name
        dest_lidar_label_dir = output_root / "interval1_LIDAR_label" / seq_name

        if not src_cam_dir.exists():
            print(f"  -> Warning: No 'interval1_CAM' found in {seq_name}. Skipping.")
            continue

        all_jpgs = sorted(list(src_cam_dir.glob("*.jpg")))
        if not all_jpgs:
            print(f"  -> Warning: No .jpg files found in {src_cam_dir}. Skipping.")
            continue
            
        selected_jpgs = all_jpgs[:args.frames_per_sequence]
        
        print(f"  -> Found {len(all_jpgs)} frames, extracting first {len(selected_jpgs)}...")

        dest_seq_dir.mkdir(parents=True, exist_ok=True)
        for item in progress_iter(list(seq_dir.iterdir()), desc=f"copy metadata {seq_name}", unit="file"):
            if item.is_file() and item.suffix in [".json", ".csv", ".png", ".txt", ".yaml"]:
                copy_file_safe(item, dest_seq_dir / item.name)

        for jpg_file in progress_iter(selected_jpgs, desc=f"copy frames {seq_name}", total=len(selected_jpgs), unit="frame"):
            frame_id = jpg_file.stem
            
            if copy_file_safe(jpg_file, dest_cam_dir / jpg_file.name):
                stats["images_copied"] += 1
                
            cam_label_src = src_cam_label_dir / f"{frame_id}.png"
            if copy_file_safe(cam_label_src, dest_cam_label_dir / f"{frame_id}.png"):
                stats["cam_labels_copied"] += 1

            lidar_src_bin = src_lidar_dir / f"{frame_id}.bin"
            lidar_src_pcd = src_lidar_dir / f"{frame_id}.pcd"
            if copy_file_safe(lidar_src_bin, dest_lidar_dir / f"{frame_id}.bin"):
                stats["lidars_copied"] += 1
            elif copy_file_safe(lidar_src_pcd, dest_lidar_dir / f"{frame_id}.pcd"):
                stats["lidars_copied"] += 1
                
            lidar_label_src = src_lidar_label_dir / f"{frame_id}.label"
            if copy_file_safe(lidar_label_src, dest_lidar_label_dir / f"{frame_id}.label"):
                stats["lidar_labels_copied"] += 1
                
        stats["sequences_processed"] += 1
        print(f"  -> Done with {seq_name}.")

    print("-" * 50)
    print("Extraction Complete!")
    print(f"Sequences Processed:  {stats['sequences_processed']}")
    print(f"RGB Images Copied:    {stats['images_copied']}")
    print(f"2D Labels Copied:     {stats['cam_labels_copied']}")
    print(f"LiDAR Scans Copied:   {stats['lidars_copied']}")
    print(f"3D Labels Copied:     {stats['lidar_labels_copied']}")
    print("-" * 50)
    print(f"The extracted subset is available at: {output_root}")


if __name__ == "__main__":
    main()
