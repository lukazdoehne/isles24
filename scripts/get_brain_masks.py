"""Get brain masks using Synthstrip"""

from pathlib import Path
import subprocess
from tqdm import tqdm


def run_synthstrip(input_path: Path, mask_path: Path, use_gpu: bool = False) -> None:
    """
    Runs the synthstrip container for brain masking.

    Parameters
    ----------
    input_path : Path
        Path to the input image (.nii or .nii.gz).
    mask_path : Path
        Path where the stripped mask should be saved.
    use_gpu : bool, optional
        If True, adds the -g flag to utilize GPU acceleration.
        Defaults is False.

    Raises
    ------
    subprocess.CalledProcessError
        If the synthstrip command fails during execution.
    """

    # Build the command list
    command = [
        "synthstrip/synthstrip-docker",
        "-i",
        str(input_path),
        "-m",
        str(mask_path),
    ]

    if use_gpu:
        command.append("-g")

    subprocess.run(command, capture_output=False, check=True)


def main():

    data_root = Path("data")
    raw_dir = data_root / "train/raw_data"
    deriv_dir = data_root / "train/derivatives"

    for case_dir_raw in tqdm(sorted(raw_dir.glob("sub-stroke*")), "Processing cases"):
        case_name = case_dir_raw.name
        case_dir_deriv = deriv_dir / case_name

        ncct_path = case_dir_raw / f"ses-01/{case_name}_ses-01_ncct.nii.gz"
        mask_path = (
            case_dir_deriv / f"ses-01/{case_name}_ses-01_space-ncct_brain-msk.nii.gz"
        )

        run_synthstrip(input_path=ncct_path, mask_path=mask_path)


if __name__ == "__main__":
    main()
