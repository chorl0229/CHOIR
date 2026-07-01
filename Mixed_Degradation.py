"""
CHOIR — Mixed Degradation Restoration
Use --Scene to select S1/S2/S3 (default S1).
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from choir_core import (
    CHOIRNet,
    CHOIRTrainer,
    DATASET_NAME,
    GAUSSIAN_STD,
    MAX_ITERATIONS,
    SALT_PEPPER_RATIO,
    SEED,
    STRUCTURAL_MISSING_RATIO,
    TARGET_SIZE,
    count_parameters,
    dataset_path,
    load_lpips,
    load_normalized_volume,
    log_final_summary,
    save_outputs,
    save_run_settings,
    setup_device,
    setup_logger,
    tensors_from_volume,
    train_and_evaluate,
    get_mgrid,
    RESULTS_DIR,
)


def apply_mixed_degradation(volume: np.ndarray, scenario: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    print(f"Mixed degradation scenario: {scenario}")
    print(f"  [S1] Gaussian noise (std={GAUSSIAN_STD})")
    observed = np.clip(volume + rng.normal(0, GAUSSIAN_STD, volume.shape).astype(np.float32), 0, 1)
    mask = np.ones_like(volume, dtype=np.float32)

    if scenario in ("S2", "S3"):
        print(f"  [S2] Salt-and-pepper noise (ratio={SALT_PEPPER_RATIO})")
        sp = rng.random(volume.shape)
        salt = sp > (1 - SALT_PEPPER_RATIO / 2)
        pepper = sp < (SALT_PEPPER_RATIO / 2)
        observed[salt] = 1.0
        observed[pepper] = 0.0
        mask[salt | pepper] = 0

    if scenario == "S3":
        h, w = volume.shape[:2]
        n_rows = max(1, int(round(h * STRUCTURAL_MISSING_RATIO)))
        n_cols = max(1, int(round(w * STRUCTURAL_MISSING_RATIO)))
        rows = rng.choice(h, min(n_rows, h), replace=False)
        cols = rng.choice(w, min(n_cols, w), replace=False)
        print(f"  [S3] Structural missing (rows={len(rows)}, cols={len(cols)})")
        observed[rows, ...] = 0
        observed[:, cols, ...] = 0
        mask[rows, ...] = 0
        mask[:, cols, ...] = 0

    return observed, mask


def prepare_data(scenario: str, device):
    loaded = load_normalized_volume()
    volume = loaded["volume"]
    observed, mask = apply_mixed_degradation(volume, scenario)
    print(f"Valid observation ratio: {np.sum(mask) / mask.size:.1%}")
    print(f"{'=' * 70}\n")
    data = tensors_from_volume(volume, observed, mask, device)
    data["original_shape_4d"] = loaded["original_shape_4d"]
    return data


def run_experiment(scenario: str = "S1"):
    scenario = scenario.upper()
    if scenario not in {"S1", "S2", "S3"}:
        raise ValueError(f"Unsupported scene: {scenario}")

    device = setup_device()
    data_name = Path(DATASET_NAME).stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = Path(RESULTS_DIR) / "mixed_degradation" / f"{timestamp}_{data_name}_{scenario}_CHOIR"
    exp_dir.mkdir(parents=True, exist_ok=True)

    save_run_settings(
        exp_dir,
        {
            "task": "mixed_degradation",
            "scene": scenario,
            "dataset": dataset_path(),
            "target_size": TARGET_SIZE,
            "max_iterations": MAX_ITERATIONS,
        },
    )
    logger = setup_logger(exp_dir)

    print(f"\n{'=' * 70}")
    print("CHOIR Mixed Degradation Experiment")
    print(f"Dataset: {DATASET_NAME} | Scene: {scenario}")
    print(f"Device: {device} | Output: {exp_dir}")
    print(f"{'=' * 70}\n")

    data_dict = prepare_data(scenario, device)
    h, w, c = data_dict["shape"]
    coords = get_mgrid((h, w)).to(device)

    model = CHOIRNet(out_features=c, signal_dims=(h, w, c)).to(device)
    model_params = count_parameters(model)
    print(f"Parameters: {model_params['total_params_m']:.4f} M")

    trainer = CHOIRTrainer(model)
    lpips_model = load_lpips(device)

    out = train_and_evaluate(
        trainer,
        coords,
        data_dict["original"].reshape(-1, c),
        data_dict["observed"].reshape(-1, c),
        data_dict["mask"].reshape(-1, c),
        h,
        w,
        c,
        lpips_model,
        logger,
        use_data_consistency=False,
    )
    log_final_summary(out["best"], model_params, logger)

    return save_outputs(
        exp_dir,
        data_name,
        scenario,
        data_dict,
        out["best"],
        model_params,
        out["history"],
        f"Observed ({scenario})",
        {"scenario": scenario},
    )


def parse_args():
    parser = argparse.ArgumentParser(description="CHOIR mixed degradation experiment.")
    parser.add_argument("--Scene", type=str, default="S1", choices=["S1", "S2", "S3"])
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_experiment(args.Scene)
    print(f"\nDone. Final PSNR: {results['best_psnr']:.4f} dB")


if __name__ == "__main__":
    main()
