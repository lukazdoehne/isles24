"""
Calculate lesion masks from logits
"""

from pathlib import Path
import re
from tqdm import tqdm
import pandas as pd
import nibabel as nib
import numpy as np
from scipy.special import softmax
from isles.metrics import (
    compute_dice_f1_instance_difference,
    compute_absolute_volume_difference,
)

def get_case(name: str) -> str:
    """Get case ID from image name"""
    return re.search(r"sub-stroke\d+", name).group(0)


def main():

    pred_root = Path("/home/renku/work/data-local/runs/run-021/logit-sweep/")
    # pred_root = Path("/home/renku/work/data-local/runs/run-021/logit-sweep/tta_flips")
    mask_root = Path("/home/renku/work/data-local/train/derivatives")

    sweep_dirs = sorted(pred_root.glob("*/"))
    case_list = sorted(get_case(i.name) for i in sweep_dirs[0].glob("logits/*nii.gz"))

    for sweep_dir in sweep_dirs:
        if (sweep_dir / "pred").exists():
            continue
        results = []
        for case in tqdm(case_list, "Processing cases"):
            logit_name = f"logits/{case}_ses-01_space-ncct_cta_logits.nii.gz"
            mask_path = mask_root / f"{case}/ses-02/{case}_ses-02_space-ncct_lesion-msk.nii.gz"
            mask = nib.load(mask_path)

            logits = nib.load(sweep_dir / logit_name).get_fdata()
            pred = softmax(logits, axis=-1)[..., 1] > 0.5

            pred_path = sweep_dir / logit_name.replace("logits", "pred")
            pred_path.parent.mkdir(exist_ok=True, parents=True)
            pred_img = nib.Nifti1Image(pred, affine=mask.affine, header=mask.header)
            nib.save(pred_img, pred_path)

            # Voxel size in mL (convert from mm^3)
            voxel_spacing = np.array(mask.header.get_zooms())
            voxel_size = np.prod(voxel_spacing) / 1000

            label = mask.get_fdata().astype(int)

            # Compute metrics
            abs_vol_diff = compute_absolute_volume_difference(
                label, pred, voxel_size
            )
            f1_score, instance_count_diff, dice_score = (
                compute_dice_f1_instance_difference(label, pred)
            )

            results.append(
                {
                    "case_id": case,
                    "dice": dice_score,
                    "f1_score": f1_score,
                    "abs_vol_diff": abs_vol_diff,
                    "instance_count_diff": instance_count_diff,
                }
            )

        results_df = pd.DataFrame(results)
        results_df.to_csv(sweep_dir / "results.csv", index=False)

if __name__ == "__main__":
    main()
