"""Shared CHOIR model, training utilities, and global hyperparameters."""

import os
import json
import math
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not installed; LPIPS metrics will be skipped. Install: pip install lpips")

try:
    from skimage.metrics import structural_similarity as ssim_skimage
    from skimage.transform import resize as skimage_resize
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not installed; SSIM and resize are unavailable.")


# ---------------------------------------------------------------------------
# Global hyperparameters (edit here to control all experiments)
# ---------------------------------------------------------------------------
HIDDEN_FEATURES = 128
NUM_HIDDEN_LAYERS = 8
LEARNING_RATE = 3.0e-3
MAX_ITERATIONS = 5000
EVAL_FREQUENCY = 100
EARLY_STOP_PATIENCE = 10
SEED = 42
GPU_ID = 0
TARGET_SIZE = 256

DATASET_NAME = "flowers.mat"
DATA_KEY = "orig"
RESULTS_DIR = "./results"

# RGB bands for hyperspectral visualization (MSI Flowers)
R_BAND, G_BAND, B_BAND = 21, 12, 5

# Mixed degradation defaults (S1/S2/S3 building blocks)
GAUSSIAN_STD = 0.2
SALT_PEPPER_RATIO = 0.1
STRUCTURAL_MISSING_RATIO = 0.03


def dataset_path() -> str:
    return str(Path(__file__).resolve().parent / "dataset" / DATASET_NAME)


def setup_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{GPU_ID}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    return device


def setup_logger(exp_dir: Path) -> logging.Logger:
    log_file = exp_dir / "experiment.log"
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized: {log_file}")
    return logger


def save_run_settings(exp_dir: Path, settings: Dict[str, Any]) -> None:
    with open(exp_dir / "run_settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


def get_mgrid(sidelen: Tuple[int, ...], dim: int = 2) -> torch.Tensor:
    """Normalized coordinate grid in [-1, 1]; 2D order is (x, y)."""
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)
    if dim == 2:
        pixel_coords = np.stack(np.mgrid[: sidelen[0], : sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] /= sidelen[0] - 1
        pixel_coords[0, :, :, 1] /= sidelen[1] - 1
    else:
        raise NotImplementedError(f"get_mgrid does not support dim={dim}")
    pixel_coords -= 0.5
    pixel_coords *= 2.0
    pixel_coords = pixel_coords[..., [1, 0]]
    return torch.from_numpy(pixel_coords.reshape(-1, dim))


class PerceptualLayer(nn.Module):
    """PSC layer: log-uniform frequencies with learnable amplitude decay."""

    def __init__(self, in_features: int, out_features: int, is_first: bool, signal_dims: Tuple[int, int, int]):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.log_alpha = nn.Parameter(torch.tensor(np.log(2.0)))

        d_min = min(signal_dims[:2])
        omega_min = np.pi
        omega_max = 0.125 * (np.pi * d_min) / 2.0
        omegas = torch.logspace(np.log10(omega_min), np.log10(omega_max), out_features, dtype=torch.float32)
        self.register_buffer("omegas", omegas)
        self._init_weights(is_first)

    def _init_weights(self, is_first: bool) -> None:
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / self.linear.in_features, 1.0 / self.linear.in_features)
            else:
                bound = np.sqrt(6.0 / self.linear.in_features) / self.omegas.mean().item()
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.log_alpha)
        geo_mean = torch.exp(torch.mean(torch.log(self.omegas + 1e-9)))
        amplitudes = 1.0 / torch.pow(self.omegas / geo_mean, alpha / 2.0)
        linear_out = self.linear(x)
        return amplitudes.unsqueeze(0) * torch.sin(self.omegas.unsqueeze(0) * linear_out)


class ReZeroBlock(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.beta * self.layer(x)


class CHOIRNet(nn.Module):
    """CHOIR MLP: perceptual sine layers with ReZero residual stacking."""

    def __init__(self, out_features: int, signal_dims: Tuple[int, int, int]):
        super().__init__()
        self.first_layer = PerceptualLayer(2, HIDDEN_FEATURES, is_first=True, signal_dims=signal_dims)
        hidden = []
        for _ in range(NUM_HIDDEN_LAYERS):
            layer = PerceptualLayer(HIDDEN_FEATURES, HIDDEN_FEATURES, is_first=False, signal_dims=signal_dims)
            hidden.append(ReZeroBlock(layer))
        self.hidden_net = nn.Sequential(*hidden)
        self.final_linear = nn.Linear(HIDDEN_FEATURES, out_features)
        with torch.no_grad():
            bound = np.sqrt(6.0 / HIDDEN_FEATURES)
            self.final_linear.weight.uniform_(-bound, bound)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(coords)
        x = self.hidden_net(x)
        return torch.sigmoid(self.final_linear(x))


class CHOIRTrainer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=MAX_ITERATIONS, eta_min=1e-6
        )
        self.criterion = nn.L1Loss()
        self.best_psnr = float("-inf")
        self.epochs_no_improve = 0

    def train_step(self, coords: torch.Tensor, observed: torch.Tensor, mask: torch.Tensor) -> float:
        self.model.train()
        prediction = self.model(coords)
        loss = self.criterion(prediction * mask, observed * mask)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def should_early_stop(self, psnr: float) -> bool:
        if psnr > self.best_psnr + 1e-4:
            self.best_psnr = psnr
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
        return self.epochs_no_improve >= EARLY_STOP_PATIENCE

    @torch.no_grad()
    def inference(self, coords: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(coords)


def load_mat_data(data_path: str, data_key: str = DATA_KEY) -> np.ndarray:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    mat_data = sio.loadmat(data_path)
    if data_key not in mat_data:
        keys = [k for k in mat_data if not k.startswith("__")]
        raise ValueError(f"Key '{data_key}' not found. Available: {keys}")
    return mat_data[data_key].astype(np.float32)


def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    data_min, data_max = np.min(data), np.max(data)
    if data_max > data_min:
        return (data - data_min) / (data_max - data_min), data_min, data_max
    return np.zeros_like(data), data_min, data_max


def resize_data(data: np.ndarray, target_size: int) -> np.ndarray:
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image is required for resizing")
    original_ndim = data.ndim
    if original_ndim == 4:
        h, w, c, d = data.shape
        data_3d = data.reshape(h, w, -1)
    elif original_ndim == 3:
        data_3d = data
    else:
        raise ValueError(f"Unsupported data rank: {original_ndim}")
    h, w = data_3d.shape[:2]
    if h == target_size and w == target_size:
        return data
    flat_c = data_3d.shape[2]
    resized = skimage_resize(
        data_3d, (target_size, target_size, flat_c), order=1, preserve_range=True, anti_aliasing=True
    ).astype(np.float32)
    if original_ndim == 4:
        try:
            return resized.reshape(target_size, target_size, c, d)
        except ValueError:
            print(f"Warning: cannot restore 4D shape; keeping 3D {resized.shape}")
    return resized


def load_normalized_volume(data_path: str = None) -> Dict[str, Any]:
    path = data_path or dataset_path()
    print(f"\n{'=' * 70}")
    print(f"Loading data: {path}")
    raw = load_mat_data(path)
    print(f"Raw shape: {raw.shape}")
    original_shape_4d = None
    if raw.ndim == 4:
        original_shape_4d = raw.shape
        raw = raw.reshape(raw.shape[0], raw.shape[1], -1)
        print(f"[Data] Reshaped to 3D: {raw.shape}")
    h, w = raw.shape[:2]
    if h != TARGET_SIZE or w != TARGET_SIZE:
        print(f"Resize: {h}x{w} -> {TARGET_SIZE}x{TARGET_SIZE}")
        raw = resize_data(raw, TARGET_SIZE)
    normalized, data_min, data_max = normalize_data(raw)
    print("Normalization complete.")
    return {
        "volume": normalized,
        "original_shape_4d": original_shape_4d,
        "data_min": data_min,
        "data_max": data_max,
    }


def tensors_from_volume(volume: np.ndarray, observed: np.ndarray, mask: np.ndarray, device: torch.device) -> Dict[str, Any]:
    h, w, c = volume.shape
    return {
        "original": torch.from_numpy(volume).float().to(device),
        "observed": torch.from_numpy(observed).float().to(device),
        "mask": torch.from_numpy(mask).float().to(device),
        "shape": (h, w, c),
    }


def format_significant(value: float, sig_digits: int = 4) -> str:
    if value < 0:
        return "N/A"
    if value == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(value)))
    decimals = max(0, sig_digits - 1 - magnitude)
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return (20 * torch.log10(torch.tensor(1.0, device=pred.device) / torch.sqrt(mse))).item()


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    if not SKIMAGE_AVAILABLE:
        return -1.0
    pred_np = np.clip(pred.detach().cpu().numpy(), 0, 1)
    target_np = np.clip(target.detach().cpu().numpy(), 0, 1)
    vals = [
        ssim_skimage(pred_np[:, :, i], target_np[:, :, i], data_range=1.0) for i in range(pred_np.shape[-1])
    ]
    return float(np.mean(vals))


def calculate_lpips(pred: torch.Tensor, target: torch.Tensor, lpips_model: Optional[Any]) -> float:
    if not LPIPS_AVAILABLE or lpips_model is None:
        return -1.0
    try:
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        _, _, c = pred_np.shape
        if c == 1:
            pred_rgb = np.repeat(pred_np, 3, axis=2)
            target_rgb = np.repeat(target_np, 3, axis=2)
        elif c == 3:
            pred_rgb, target_rgb = pred_np, target_np
        else:
            idx = [min(i, c - 1) for i in [0, c // 2, c - 1]]
            pred_rgb = pred_np[:, :, idx]
            target_rgb = target_np[:, :, idx]
        pred_t = torch.from_numpy(pred_rgb).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        target_t = torch.from_numpy(target_rgb).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        dev = next(lpips_model.parameters()).device
        with torch.no_grad():
            return float(lpips_model(pred_t.to(dev), target_t.to(dev)).item())
    except Exception as e:
        print(f"Warning: LPIPS failed: {e}")
        return -1.0


def count_parameters(model: nn.Module) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    return {"total_params": total, "total_params_m": total / 1e6}


def load_lpips(device: torch.device):
    if not LPIPS_AVAILABLE:
        return None
    try:
        model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        print("LPIPS loaded (backbone: alex)")
        return model
    except Exception as e:
        print(f"Warning: LPIPS load failed: {e}")
        return None


def hyperspectral_to_rgb(data: np.ndarray) -> np.ndarray:
    h, w, c = data.shape
    r_idx = min(R_BAND, c - 1)
    g_idx = min(G_BAND, c - 1)
    b_idx = min(B_BAND, c - 1)
    rgb = np.stack([data[:, :, r_idx], data[:, :, g_idx], data[:, :, b_idx]], axis=-1)
    for i in range(3):
        ch = rgb[:, :, i]
        lo, hi = np.percentile(ch, [2, 98])
        if hi > lo:
            rgb[:, :, i] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return rgb


def save_comparison_figure(
    original: np.ndarray,
    observed: np.ndarray,
    reconstruction: np.ndarray,
    psnr: float,
    ssim: float,
    lpips_val: float,
    observed_title: str,
    save_path: str,
) -> None:
    original_rgb = hyperspectral_to_rgb(original)
    observed_rgb = hyperspectral_to_rgb(observed)
    recon_rgb = hyperspectral_to_rgb(reconstruction)
    error = np.sqrt(np.sum((original - reconstruction) ** 2, axis=-1))
    error_norm = (error - error.min()) / (error.max() - error.min() + 1e-8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Ground Truth", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(observed_rgb)
    axes[1].set_title(observed_title, fontsize=14, fontweight="bold")
    axes[1].axis("off")
    title = f"CHOIR Reconstruction\nPSNR: {psnr:.2f}dB, SSIM: {ssim:.4f}"
    if lpips_val >= 0:
        title += f", LPIPS: {format_significant(lpips_val, 4)}"
    axes[2].imshow(recon_rgb)
    axes[2].set_title(title, fontsize=14, fontweight="bold")
    axes[2].axis("off")
    im = axes[3].imshow(error_norm, cmap="hot")
    axes[3].set_title("Reconstruction Error", fontsize=14, fontweight="bold")
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Visualization saved: {save_path}")


def save_training_curves(history: Dict[str, Any], save_path: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].plot(history["losses"])
    axes[0].set_title("Training Loss", fontsize=12, fontweight="bold")
    axes[0].set_yscale("log")
    axes[0].grid(True)

    eval_iters = np.arange(EVAL_FREQUENCY, len(history["psnr"]) * EVAL_FREQUENCY + 1, EVAL_FREQUENCY)
    axes[1].plot(eval_iters, history["eval_losses"], color="red")
    axes[1].set_title("Evaluation Loss", fontsize=12, fontweight="bold")
    axes[1].set_yscale("log")
    axes[1].grid(True)
    axes[2].plot(eval_iters, history["psnr"])
    axes[2].set_title("PSNR History", fontsize=12, fontweight="bold")
    axes[2].grid(True)
    axes[3].plot(eval_iters, history["ssim"])
    axes[3].set_title("SSIM History", fontsize=12, fontweight="bold")
    axes[3].grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def train_and_evaluate(
    trainer: CHOIRTrainer,
    coords: torch.Tensor,
    original_flat: torch.Tensor,
    observed_flat: torch.Tensor,
    mask_flat: torch.Tensor,
    h: int,
    w: int,
    c: int,
    lpips_model,
    logger: logging.Logger,
    use_data_consistency: bool,
) -> Dict[str, Any]:
    history = {"losses": [], "eval_losses": [], "psnr": [], "ssim": [], "lpips": []}
    best = {
        "loss": float("inf"),
        "psnr": 0.0,
        "ssim": 0.0,
        "lpips": float("inf"),
        "reconstruction": None,
        "iteration": 0,
        "state": None,
        "runtime": 0.0,
    }
    peak_psnr = 0.0
    train_time = 0.0

    print(f"\n{'=' * 70}\nTraining started.\n{'=' * 70}\n")
    for iteration in range(MAX_ITERATIONS):
        t0 = time.time()
        loss = trainer.train_step(coords, observed_flat, mask_flat)
        train_time += time.time() - t0
        history["losses"].append(loss)

        if (iteration + 1) % EVAL_FREQUENCY != 0:
            continue

        with torch.no_grad():
            pred = trainer.inference(coords).reshape(h, w, c)
            original = original_flat.reshape(h, w, c)
            observed = observed_flat.reshape(h, w, c)
            mask = mask_flat.reshape(h, w, c)
            recon = mask * observed + (1 - mask) * pred if use_data_consistency else pred

            eval_loss = trainer.criterion(pred.reshape(-1, c) * mask_flat, observed_flat * mask_flat).item()
            psnr = calculate_psnr(recon, original)
            ssim = calculate_ssim(recon, original)
            lp = calculate_lpips(recon, original, lpips_model)

            history["eval_losses"].append(eval_loss)
            history["psnr"].append(psnr)
            history["ssim"].append(ssim)
            history["lpips"].append(lp)

            if psnr > peak_psnr:
                peak_psnr = psnr
                best["runtime"] = train_time

            if psnr > best["psnr"]:
                best.update(
                    loss=eval_loss,
                    psnr=psnr,
                    ssim=ssim,
                    lpips=lp if lp >= 0 else best["lpips"],
                    reconstruction=recon.clone(),
                    iteration=iteration + 1,
                    state={k: v.cpu().clone() for k, v in trainer.model.state_dict().items()},
                )

            marker = " [BEST]" if psnr == best["psnr"] else ""
            lp_str = format_significant(lp, 4) if lp >= 0 else "N/A"
            msg = (
                f"Iter {iteration + 1:5d}/{MAX_ITERATIONS} | "
                f"Train Loss: {loss:.6f} | Eval Loss: {eval_loss:.6f}{marker} | "
                f"PSNR: {psnr:.2f}dB | SSIM: {ssim:.4f} | LPIPS: {lp_str}"
            )
            print(msg)
            logger.info(msg)

            if trainer.should_early_stop(psnr):
                stop_msg = f"\nEarly stopping at iteration {iteration + 1}"
                print(stop_msg)
                logger.info(stop_msg)
                break

    if best["reconstruction"] is None:
        with torch.no_grad():
            pred = trainer.inference(coords).reshape(h, w, c)
            original = original_flat.reshape(h, w, c)
            observed = observed_flat.reshape(h, w, c)
            mask = mask_flat.reshape(h, w, c)
            recon = mask * observed + (1 - mask) * pred if use_data_consistency else pred
            best.update(
                loss=trainer.criterion(pred.reshape(-1, c) * mask_flat, observed_flat * mask_flat).item(),
                psnr=calculate_psnr(recon, original),
                ssim=calculate_ssim(recon, original),
                lpips=calculate_lpips(recon, original, lpips_model),
                reconstruction=recon.clone(),
                iteration=MAX_ITERATIONS,
                state={k: v.cpu().clone() for k, v in trainer.model.state_dict().items()},
                runtime=train_time,
            )

    if best["runtime"] == 0.0:
        best["runtime"] = train_time

    return {"history": history, "best": best}


def log_final_summary(best: Dict[str, Any], model_params: Dict[str, float], logger: logging.Logger) -> None:
    lines = [
        f"\n{'=' * 70}",
        "Training complete.",
        f"{'=' * 70}",
        f"Best @ iter {best['iteration']}:",
        f"  - Loss: {best['loss']:.6f}",
        f"  - PSNR: {best['psnr']:.4f} dB",
        f"  - SSIM: {best['ssim']:.4f}",
    ]
    if best["lpips"] < float("inf"):
        lines.append(f"  - LPIPS: {format_significant(best['lpips'], 4)}")
    lines.extend(
        [
            f"  - Runtime: {best['runtime']:.2f}s",
            f"  - Parameters: {model_params['total_params_m']:.4f} M",
            f"{'=' * 70}\n",
        ]
    )
    for line in lines:
        print(line)
        logger.info(line)


def save_outputs(
    exp_dir: Path,
    data_name: str,
    tag: str,
    data_dict: Dict[str, Any],
    best: Dict[str, Any],
    model_params: Dict[str, float],
    history: Dict[str, Any],
    observed_title: str,
    results_extra: Dict[str, Any],
) -> Dict[str, Any]:
    h, w, c = data_dict["shape"]
    comparison_path = exp_dir / f"comparison_{data_name}_{tag}.png"
    save_comparison_figure(
        data_dict["original"].cpu().numpy(),
        data_dict["observed"].cpu().numpy(),
        best["reconstruction"].cpu().numpy(),
        best["psnr"],
        best["ssim"],
        best["lpips"] if best["lpips"] < float("inf") else -1.0,
        observed_title,
        str(comparison_path),
    )

    curves_path = exp_dir / f"training_curves_{data_name}_{tag}.png"
    save_training_curves(history, str(curves_path))
    print(f"Training curves saved: {curves_path}")

    model_path = exp_dir / f"model_{data_name}_{tag}.pth"
    torch.save(
        {
            "model_state_dict": best["state"],
            "data_shape": {"h": h, "w": w, "c": c},
            "best_psnr": best["psnr"],
            "best_ssim": best["ssim"],
            "iteration": best["iteration"],
            "model_params": model_params,
        },
        model_path,
    )
    print(f"Model saved (iter {best['iteration']}): {model_path}")

    results = {
        "data_name": data_name,
        "best_loss": best["loss"],
        "best_psnr": best["psnr"],
        "best_ssim": best["ssim"],
        "best_lpips": best["lpips"] if best["lpips"] < float("inf") else -1.0,
        "best_iteration": best["iteration"],
        "total_time": best["runtime"],
        "total_params": model_params["total_params"],
        "total_params_m": model_params["total_params_m"],
        **results_extra,
    }
    results_path = exp_dir / f"results_{data_name}_{tag}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved: {results_path}")

    recon_np = best["reconstruction"].cpu().numpy()
    original_np = data_dict["original"].cpu().numpy()
    shape_4d = data_dict.get("original_shape_4d")
    if shape_4d is not None:
        try:
            recon_np = recon_np.reshape(shape_4d)
            original_np = original_np.reshape(shape_4d)
        except ValueError as e:
            print(f"Warning: cannot restore 4D shape ({e}); saving as 3D.")

    mat_path = exp_dir / f"reconstruction_{data_name}_{tag}.mat"
    mat_dict = {
        "reconstruction": recon_np,
        "original": original_np,
        "loss": best["loss"],
        "psnr": best["psnr"],
        "ssim": best["ssim"],
        "lpips": best["lpips"] if best["lpips"] < float("inf") else -1.0,
        "best_iteration": best["iteration"],
    }
    if shape_4d is not None:
        mat_dict["original_shape_4d"] = np.array(shape_4d)
    sio.savemat(str(mat_path), mat_dict)
    print(f"Reconstruction saved (iter {best['iteration']}): {mat_path}")
    return results
