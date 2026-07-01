"""
CHOIR — Random Missing Data Completion
Use --OR to set the observation ratio (default 0.10).
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from choir_core import (
    CHOIRNet,
    CHOIRTrainer,
    DATASET_NAME,
    MAX_ITERATIONS,
    SEED,
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


def generate_random_mask(shape, ratio: float, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = np.prod(shape)
    idx = rng.choice(total, int(total * ratio), replace=False)
    mask = np.zeros(total, dtype=np.float32)
    mask[idx] = 1
    return mask.reshape(shape)


def prepare_data(observation_ratio: float, device):
    loaded = load_normalized_volume()
    volume = loaded["volume"]
    print(f"Random mask: OR={observation_ratio:.1%}, seed={SEED}")
    mask = generate_random_mask(volume.shape, observation_ratio)
    observed = volume * mask
    print(f"Effective OR: {np.sum(mask) / mask.size:.1%}")
    print(f"{'=' * 70}\n")
    data = tensors_from_volume(volume, observed, mask, device)
    data["original_shape_4d"] = loaded["original_shape_4d"]
    data["data_min"] = loaded["data_min"]
    data["data_max"] = loaded["data_max"]
    return data


def run_experiment(observation_ratio: float = 0.10):
    device = setup_device()
    data_name = Path(DATASET_NAME).stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = Path(RESULTS_DIR) / "random_missing" / f"{timestamp}_{data_name}_SR{observation_ratio}_CHOIR"
    exp_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "task": "random_missing",
        "OR": observation_ratio,
        "dataset": dataset_path(),
        "target_size": TARGET_SIZE,
        "max_iterations": MAX_ITERATIONS,
    }
    save_run_settings(exp_dir, settings)
    logger = setup_logger(exp_dir)

    print(f"\n{'=' * 70}")
    print("CHOIR Random Missing Experiment")
    print(f"Dataset: {DATASET_NAME} | OR: {observation_ratio:.1%}")
    print(f"Device: {device} | Output: {exp_dir}")
    print(f"{'=' * 70}\n")
    logger.info(f"Dataset: {DATASET_NAME} | OR: {observation_ratio:.1%}")

    data_dict = prepare_data(observation_ratio, device)
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
        use_data_consistency=True,
    )
    log_final_summary(out["best"], model_params, logger)

    tag = f"SR{observation_ratio}"
    return save_outputs(
        exp_dir,
        data_name,
        tag,
        data_dict,
        out["best"],
        model_params,
        out["history"],
        f"Observed (SR: {observation_ratio:.1%})",
        {"sampling_ratio": observation_ratio},
    )


def parse_args():
    parser = argparse.ArgumentParser(description="CHOIR random missing experiment.")
    parser.add_argument("--OR", type=float, default=0.10, help="Observation ratio in (0, 1].")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.OR <= 1:
        raise ValueError(f"--OR must be in (0, 1], got {args.OR}")
    results = run_experiment(args.OR)
    print(f"\nDone. Final PSNR: {results['best_psnr']:.4f} dB")


if __name__ == "__main__":
    main()
