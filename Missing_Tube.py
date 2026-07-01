"""
CHOIR — Tube Missing Data Completion
Use --OR to set the tube observation ratio (default 0.10).
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


def generate_tube_mask(shape, ratio: float, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w, c = shape
    total_tubes = h * w
    idx = rng.choice(total_tubes, int(total_tubes * ratio), replace=False)
    spatial = np.zeros(total_tubes, dtype=np.float32)
    spatial[idx] = 1
    spatial = spatial.reshape(h, w)
    return np.repeat(spatial[:, :, np.newaxis], c, axis=2)


def prepare_data(observation_ratio: float, device):
    loaded = load_normalized_volume()
    volume = loaded["volume"]
    print(f"[Tube Missing] OR={observation_ratio:.1%}, seed={SEED}")
    mask = generate_tube_mask(volume.shape, observation_ratio)
    observed = volume * mask
    tubes = int(np.sum(mask[:, :, 0]))
    print(f"[Tube Missing] Observed tubes: {tubes}/{volume.shape[0] * volume.shape[1]}")
    print(f"Effective OR: {np.sum(mask) / mask.size:.1%}")
    print(f"{'=' * 70}\n")
    data = tensors_from_volume(volume, observed, mask, device)
    data["original_shape_4d"] = loaded["original_shape_4d"]
    return data


def run_experiment(observation_ratio: float = 0.10):
    device = setup_device()
    data_name = Path(DATASET_NAME).stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    scenario = "tube_missing"
    exp_dir = (
        Path(RESULTS_DIR)
        / "non_uniform_missing"
        / f"{timestamp}_{data_name}_{scenario}_ratio{observation_ratio}_CHOIR"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    save_run_settings(
        exp_dir,
        {
            "task": "tube_missing",
            "OR": observation_ratio,
            "dataset": dataset_path(),
            "target_size": TARGET_SIZE,
            "max_iterations": MAX_ITERATIONS,
        },
    )
    logger = setup_logger(exp_dir)

    print(f"\n{'=' * 70}")
    print(f"CHOIR Tube Missing Experiment ({scenario})")
    print(f"Dataset: {DATASET_NAME} | OR: {observation_ratio:.1%}")
    print(f"Device: {device} | Output: {exp_dir}")
    print(f"{'=' * 70}\n")

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

    return save_outputs(
        exp_dir,
        data_name,
        scenario,
        data_dict,
        out["best"],
        model_params,
        out["history"],
        f"Observed (Tube Missing, OR: {observation_ratio:.1%})",
        {"scenario": scenario, "observation_ratio": observation_ratio},
    )


def parse_args():
    parser = argparse.ArgumentParser(description="CHOIR tube missing experiment.")
    parser.add_argument("--OR", type=float, default=0.10, help="Tube observation ratio in (0, 1].")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.OR <= 1:
        raise ValueError(f"--OR must be in (0, 1], got {args.OR}")
    results = run_experiment(args.OR)
    print(f"\nDone. Final PSNR: {results['best_psnr']:.4f} dB")


if __name__ == "__main__":
    main()
