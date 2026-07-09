# Point Cloud Dataset Workflow

This document describes the end-to-end workflow for generating point cloud training data on the HPC cluster and using it for neural field training locally.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     HPC CLUSTER (Imperial)                       │
├─────────────────────────────────────────────────────────────────┤
│  1. Generate reference mesh (reference_avg.msh)                  │
│  2. Run PBS job: 128 electrode positions × FEM solve             │
│  3. Output: 128 point cloud samples (.npz files)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ rsync / scp
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LOCAL MACHINE                                │
├─────────────────────────────────────────────────────────────────┤
│  4. Load point cloud data                                        │
│  5. Train neural field: u(x,y,z | electrode_position)            │
│  6. Evaluate and visualize                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Step 1: Generate Reference Mesh (Cluster)

```bash
# SSH to cluster
ssh <user>@<hpc-login-node>

cd /path/to/emgforge
conda activate fenicsx-env
export PYTHONPATH=src

# Generate mesh with average layer dimensions
python scripts/generate_reference_mesh.py --verbose
```

Output:
- `meshes/reference_avg.msh` - FEM mesh (781K nodes, 4.4M elements)
- `metadata/reference_avg.json` - Geometry parameters

## Step 2: Generate Point Cloud Data (Cluster)

```bash
cd pbs_scripts/pointcloud
./submit_electrode.sh
```

Monitor progress:
```bash
qstat -u <user>
tail -f ${DATA_ROOT:-./data}/pointcloud_data/electrode_sweep_128/logs/reference_avg.log
```

## Step 3: Transfer Data to Local Machine

```bash
# On local machine
rsync -avz --progress \
    <user>@<hpc-login-node>:${DATA_ROOT:-./data}/pointcloud_data/electrode_sweep_128/reference_avg/ \
    ./data/pointcloud_electrode_128/
```

Expected files:
```
data/pointcloud_electrode_128/
├── elec_000000.npz
├── elec_000001.npz
├── ...
├── elec_000127.npz
└── manifest.json
```

## Step 4: Data Format

Each `.npz` file contains:

| Array | Shape | Description |
|-------|-------|-------------|
| `points` | (N, 3) | Query point coordinates (x, y, z) in mm |
| `u` | (N,) | Electric potential values |
| `sigma` | (N, 3, 3) | Conductivity tensor at each point |
| `tissue_ids` | (N,) | Tissue type (0=canc, 1=cort, 2=muscle, 3=fat, 4=skin) |
| `source_position` | (3,) | Electrode position (x, y, z) |
| `electrode_cyl` | (3,) | Electrode in cylindrical coords (r_norm, theta_deg, z_norm) |
| `ground_position` | (3,) | Ground electrode position |
| `boundary_points` | (M, 3) | Boundary points for PINN training |
| `boundary_normals` | (M, 3) | Outward normals at boundary |

Default: N=10,000 interior points, M=2,000 boundary points

## Step 5: Load Data (Python)

```python
import numpy as np
import json
from pathlib import Path

# Load manifest
data_dir = Path("./data/pointcloud_electrode_128")
with open(data_dir / "manifest.json") as f:
    manifest = json.load(f)

print(f"Dataset: {manifest['dataset_type']}")
print(f"Samples: {manifest['n_samples']}")
print(f"Electrode layout: {manifest['electrode_layout']}")

# Load all samples
samples = []
for entry in manifest["samples"]:
    data = np.load(data_dir / entry["file"])
    samples.append({
        "points": data["points"],           # (N, 3)
        "u": data["u"],                     # (N,)
        "sigma": data["sigma"],             # (N, 3, 3)
        "electrode_pos": data["source_position"],  # (3,)
        "electrode_cyl": data["electrode_cyl"],    # (3,) - r_norm, theta, z_norm
    })

# Stack for batched training
all_points = np.stack([s["points"] for s in samples])      # (128, N, 3)
all_u = np.stack([s["u"] for s in samples])                # (128, N)
all_electrodes = np.stack([s["electrode_pos"] for s in samples])  # (128, 3)
```

## Step 6: Neural Field Training

The goal is to learn a function:
```
u = f(x, y, z, electrode_x, electrode_y, electrode_z)
```

### Example PyTorch Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader

class PointCloudDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        with open(self.data_dir / "manifest.json") as f:
            self.manifest = json.load(f)
        self.samples = self.manifest["samples"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.data_dir / self.samples[idx]["file"])
        return {
            "points": torch.from_numpy(data["points"]).float(),
            "u": torch.from_numpy(data["u"]).float(),
            "electrode": torch.from_numpy(data["source_position"]).float(),
        }

# Usage
dataset = PointCloudDataset("./data/pointcloud_electrode_128")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

for batch in loader:
    points = batch["points"]      # (B, N, 3)
    u = batch["u"]                # (B, N)
    electrode = batch["electrode"]  # (B, 3)

    # Expand electrode to match points
    electrode_expanded = electrode.unsqueeze(1).expand(-1, points.shape[1], -1)  # (B, N, 3)

    # Concatenate for network input
    x = torch.cat([points, electrode_expanded], dim=-1)  # (B, N, 6)

    # Forward pass: model(x) -> predicted u
```

### Example Network Architecture

```python
import torch.nn as nn

class NeuralField(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=8):
        super().__init__()
        layers = [nn.Linear(6, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, N, 6) = [x, y, z, elec_x, elec_y, elec_z]
        return self.net(x).squeeze(-1)  # (B, N)
```

## Electrode Grid Layout

The default 8×16 grid places electrodes at:
- **Z positions**: 8 evenly spaced from z_norm=0.1 to z_norm=0.9
- **Theta positions**: 16 evenly spaced from 0° to 337.5° (22.5° spacing)

```
Theta (degrees):  0   22.5  45  67.5  90  112.5 135 157.5 180 202.5 225 247.5 270 292.5 315 337.5
                  │    │    │    │    │    │    │    │    │    │    │    │    │    │    │    │
Z=0.9  ──────────●────●────●────●────●────●────●────●────●────●────●────●────●────●────●────●
Z=0.8  ──────────●────●────●────●────●────●────●────●────●────●────●────●────●────●────●────●
Z=0.7  ──────────●────●────●────●────●────●────●────●────●────●────●────●────●────●────●────●
Z=0.6  ──────────●────●────●────●────●────●────●────●────●────●────●────●────●────●────●────●
Z=0.5  ──────────●────●────●────●────●────●────●────●────●────●────●────●────●────●────●────●
Z=0.4  ──────────●────●────●────●────●────●────●────●────●────●────●────●────●────●────●────●
Z=0.3  ──────────●────●────●────●────●────●────●────●────●────●────●────●────●────●────●────●
Z=0.1  ──────────●────●────●────●────●────●────●────●────●────●────●────●────●────●────●────●
```

## Configuration Options

Generate different electrode densities:

```bash
# Dense: 128 = 8×16 (default)
./submit_electrode.sh

# Medium: 64 = 8×8
N_ELECTRODES=64 GRID_Z=8 GRID_THETA=8 ./submit_electrode.sh

# Sparse: 32 = 4×8
N_ELECTRODES=32 GRID_Z=4 GRID_THETA=8 ./submit_electrode.sh

# Random sampling instead of grid
ELECTRODE_LAYOUT=random N_ELECTRODES=100 ./submit_electrode.sh
```

## Troubleshooting

**Job fails with "missing mesh/meta":**
```bash
# Generate the reference mesh first
python scripts/generate_reference_mesh.py --verbose
```

**Out of memory:**
- Reduce `N_POINTS` (default: 10000)
- Reduce `N_BOUNDARY` (default: 2000)

**Check job logs:**
```bash
cat ${DATA_ROOT:-./data}/pointcloud_data/electrode_sweep_128/logs/reference_avg.log
```
