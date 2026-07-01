# Calibrated Harmonic Overlaid Implicit Neural Representations for Multi-Dimensional Data (CHOIR)

Official implementation of "Calibrated Harmonic Overlaid Implicit Neural Representations for Multi-Dimensional Data," ECCV 2026

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Random missing (default OR=0.10)
python Missing_Random.py
python Missing_Random.py --OR 0.30

# Tube missing (default OR=0.10)
python Missing_Tube.py
python Missing_Tube.py --OR 0.30

# Mixed degradation (default Scene=S1)
python Mixed_Degradation.py
python Mixed_Degradation.py --Scene S3
```

**`--OR`**: observation ratio in `(0, 1]`.  
**`--Scene`**: `S1` (Gaussian noise, σ=0.20), `S2` (+ salt-and-pepper, 10%), `S3` (+ structural missing, 3% rows/cols).

## Citation

```bibtex
@article{chen2026calibrated,
  title={Calibrated Harmonic Overlaid Implicit Neural Representations for Multi-Dimensional Data},
  author={Chen, Honghang and Zhang, Xiujun and Sun, Xiaoli and Xiao, Mingqing},
  journal={arXiv preprint arXiv:2606.26763},
  year={2026}
}
```
