import argparse
import json
import sys
import shutil
from pathlib import Path
from typing import Any
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emerguav_bench.degradation.image_filters import apply_degradation
from emerguav_bench.utils.progress import progress_iter


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Apply observation-degradation augmentations physically to expanded RGB samples.")
    parser.add_argument("--samples-in", required=True, help="Input path for expanded samples JSON")
    parser.add_argument("--samples-out", required=True, help="Output path for augmented samples JSON")
    parser.add_argument("--images-out-dir", required=True, help="Directory to save augmented images")
    args = parser.parse_args()

    print(f"Loading samples from {args.samples_in}...")
    samples = load_json(args.samples_in)
    
    out_dir = Path(args.images_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Applying degradations to {len(samples)} samples. Saving images to {out_dir}...")
    
    processed_cache = {}

    for sample in progress_iter(samples, desc="apply degradations", total=len(samples), unit="sample"):
        src_path = sample["rgb_frame_path"]
        
        if src_path.startswith("placeholder://") or not Path(src_path).exists():
            continue
            
        primary_degradation = sample["observation_protocol_gt"]["primary_degradation"]
        severity = sample["degradation_severity"]
        
        if primary_degradation == "clean_visual":
            continue
            
        cache_key = f"{src_path}_{primary_degradation}_{severity}"
        
        safe_sample_id = sample["sample_id"].replace("::", "_").replace("/", "_")
        dest_path = out_dir / f"{safe_sample_id}.jpg"
        
        if cache_key not in processed_cache:
            try:
                augmented_image = apply_degradation(src_path, primary_degradation, severity)
                cv2.imwrite(str(dest_path), augmented_image)
                processed_cache[cache_key] = str(dest_path)
            except Exception as e:
                print(f"Error processing {src_path}: {e}")
                continue
        else:
            cached_path = processed_cache[cache_key]
            if cached_path != str(dest_path):
                shutil.copy(cached_path, dest_path)
                
        sample["rgb_frame_path"] = str(dest_path.resolve())
        
    print(f"Finished processing. Saving updated JSON to {args.samples_out}...")
    save_json(args.samples_out, samples)
    print("Done!")


if __name__ == "__main__":
    main()
