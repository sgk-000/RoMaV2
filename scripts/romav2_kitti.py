from romav2 import RoMaV2
import cv2
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

OVERLAP_THRESHOLD = 0.8
OVERLAP_PERCENTAGE = 0.8
GPU_ID = 1
SEQUENCE_LENGTH = 5

device = torch.device(f"cuda:{GPU_ID}")

data_path = Path("/home/kobayashi/dataset/kitti/raw_formatted_dataset")

roma_model = RoMaV2()

drive_paths = sorted(data_path.glob("2011*"))
for drive_path in tqdm(drive_paths, desc=f"drive folders"):
    image_paths = sorted(drive_path.glob("*.png"))
    for i in tqdm(range(len(image_paths)), desc="ref images"):
        for j in range(len(image_paths)):
            if i == j or abs(i - j) > SEQUENCE_LENGTH:
                continue

            image_path0 = image_paths[i]
            image_path1 = image_paths[j]
            save_dir = (
                Path("/home/kobayashi/dataset/kitti/romav2_matches") / drive_path.name
            )
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{image_path0.stem}_{image_path1.stem}_matches.npz"
            inverse_save_path = (
                save_dir / f"{image_path1.stem}_{image_path0.stem}_matches.npz"
            )
            if save_path.exists():
                continue
            if inverse_save_path.exists():
                # save empty file to indicate already processed
                np.savez(
                    save_path,
                    empty_array=np.array([]),
                )

            H_A, W_A = cv2.imread(str(image_path0)).shape[:2]
            H_B, W_B = cv2.imread(str(image_path1)).shape[:2]
            # Match
            preds = roma_model.match(image_path0, image_path1)
            matches, overlaps, precision_AB, precision_BA = roma_model.sample(preds, 10000)
            if len(overlaps[overlaps > OVERLAP_THRESHOLD]) > OVERLAP_PERCENTAGE * len(overlaps):
                np.savez(
                    save_path,
                    matches=matches.detach().cpu().numpy(),
                    overlaps=overlaps.detach().cpu().numpy(),
                    precision_AB=precision_AB.detach().cpu().numpy(),
                    precision_BA=precision_BA.detach().cpu().numpy(),
                )
            else:
                # save empty file to indicate already processed
                np.savez(
                    save_path,
                    empty_array=np.array([]),
                )
            
