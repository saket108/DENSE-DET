# DENSE_DET: Quick Block Evaluation Workflow

## Overview
This guide shows how to evaluate each block in **seconds** during development. The workflow supports:
- **Quick eval**: Evaluate blocks in 10-30 seconds
- **Rapid iteration**: Test architectural changes faster
- **Loss breakdown**: See which components contribute to training
- **Memory profiling**: Monitor GPU/CPU usage per batch

---

## Quick Start (30 seconds)

### 1. Quick Evaluation on 10 Batches
```bash
python quick_eval_dense.py --checkpoint last --max_batches 10
```

Or test script functionality without data:
```bash
python quick_eval_dense.py --dry_run
```

Output shows:
```
Loss Breakdown (Average):
  Detection:  X.XXXXXX
  Quality:    X.XXXXXX
  Auxiliary:  X.XXXXXX
  ────────────────────
  TOTAL:      X.XXXXXX

Throughput:
  Avg batch time:  XX.XX ms
  Throughput:      XXX.X images/sec
  Total eval time: X.XX sec
```

### 2. Quick Training (5 epochs)
```bash
python train_dense.py --config configs/dense_det_quick.yaml --epochs 5
```

Then eval immediately:
```bash
python quick_eval_dense.py --checkpoint last --max_batches 10
```

---

## Workflow for Block Optimization

### Step 1: Modify a Block
Edit a block in `model/dense_blocks.py` or `model/dense_detector.py`:

```python
class MyImprovedBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Your changes here
        
    def forward(self, x):
        # Optimized forward pass
        return x
```

### Step 2: Quick Test (10-20 seconds)
```bash
# Option A: Evaluate a trained checkpoint
python quick_eval_dense.py \
    --checkpoint last \
    --max_batches 10 \
    --batch 16

# Option B: Fresh model with minimal training
python train_dense.py \
    --config configs/dense_det_quick.yaml \
    --epochs 3 \
    --batch 8

# Then eval immediately
python quick_eval_dense.py --checkpoint last --max_batches 10
```

### Step 3: Analyze Results
Check output for:
- **Loss increasing?** → Block may be breaking gradient flow
- **Throughput down?** → Block may be computationally expensive
- **Memory spike?** → Consider gradient checkpointing
- **Loss not decreasing?** → Architecture may need tuning

### Step 4: Iterate or Train Full
- ❌ Loss got worse? → Revert and try different approach
- ✓ Loss improved? → Train with full config:

```bash
python train_dense.py --config configs/dense_det.yaml --epochs 80
```

---

## Common Workflows

### A. Batch Size Flexibility Testing
Test if your block supports different batch sizes:

```bash
# Test batch size 4
python quick_eval_dense.py --checkpoint last --batch 4 --max_batches 20

# Test batch size 16
python quick_eval_dense.py --checkpoint last --batch 16 --max_batches 20

# Test batch size 32
python quick_eval_dense.py --checkpoint last --batch 32 --max_batches 10
```

Expected: Loss should be stable across different batch sizes (slight variance is normal).

### B. Memory Profiling
Test gradient checkpointing and memory efficiency:

```bash
# Baseline memory
python quick_eval_dense.py \
    --checkpoint last \
    --batch 32 \
    --max_batches 5

# With gradient checkpointing
python quick_eval_dense.py \
    --checkpoint last \
    --batch 32 \
    --max_batches 5
    # (after enabling in model)

# Compare GPU memory usage in output
```

### C. Architecture Ablation
Test component importance:

```bash
# With detail branch
python quick_eval_dense.py \
    --checkpoint last \
    --detail_branch \
    --max_batches 10

# Without detail branch
python quick_eval_dense.py \
    --checkpoint last \
    --no_detail_branch \
    --max_batches 10

# Compare loss breakdown
```

### D. Head Comparison
Test different head types:

```bash
# Standard head
python quick_eval_dense.py \
    --checkpoint last \
    --head_type dense \
    --max_batches 10

# Evidential head
python quick_eval_dense.py \
    --checkpoint last \
    --head_type evidential \
    --max_batches 10
```

---

## Advanced Usage

### Custom Batch Sizing
```bash
python quick_eval_dense.py \
    --checkpoint last \
    --batch 32 \
    --imgsz 512 \
    --max_batches 10
```

### Backbone Switching
```bash
# Test different backbones
python quick_eval_dense.py \
    --checkpoint last \
    --backbone_name prism_custom \
    --backbone_dims "16,32,64,128" \
    --backbone_depths "2,2,4,2" \
    --max_batches 20
```

### Loss Analysis
```bash
# Detailed loss output automatically shown
python quick_eval_dense.py \
    --checkpoint last \
    --quality_loss_weight 1.5 \
    --auxiliary_loss_weight 0.5 \
    --max_batches 10
```

### Multiple Block Testing
```bash
#!/bin/bash
# Test a series of block configurations

for batch_size in 4 8 16 32; do
    echo "Testing batch size: $batch_size"
    python quick_eval_dense.py \
        --checkpoint last \
        --batch $batch_size \
        --max_batches 5
    echo "---"
done
```

---

## Quick Eval Arguments

```bash
python quick_eval_dense.py --help

options:
  --config CONFIG              YAML config file
  --checkpoint CHECKPOINT      Path to checkpoint ('last', 'best', or path)
  --max_batches N              Batches to evaluate (default: 10)
  --batch N                    Batch size (default: 8)
  --imgsz N                    Input image size (default: 640)
  --device DEVICE              cuda or cpu (default: cuda)
  --channels_last              Use channels_last memory format
  --fuse_bn                    Fuse BatchNorm for inference speed
  
Model Options:
  --variant {tiny,small,base}
  --backbone_name {vst,prism}
  --detail_branch / --no_detail_branch
  --quality_head / --no_quality_head
  --auxiliary_heads / --no_auxiliary_heads
```

---

## Performance Targets

### Quick Eval Timing
- **10 batches, batch_size=8** → ~5-10 seconds
- **20 batches, batch_size=16** → ~10-20 seconds
- **50 batches, batch_size=4** → ~15-25 seconds

### Memory Usage (for reference)
- **Batch size 8**: ~2-3 GB
- **Batch size 16**: ~4-5 GB
- **Batch size 32**: ~8-9 GB

---

## Troubleshooting

### ❌ "Checkpoint not found"
```bash
# List available checkpoints
ls -la runs/dense_det/

# Use 'best' or 'last' which auto-resolve
python quick_eval_dense.py --checkpoint last
```

### ❌ CUDA out of memory
```bash
# Reduce batch size
python quick_eval_dense.py --batch 4 --max_batches 20

# Or use CPU (slow)
python quick_eval_dense.py --device cpu --max_batches 5
```

### ❌ Data not found
```bash
# Test script without data first
python quick_eval_dense.py --dry_run

# Then set up data paths:
# Option 1: Environment variable
set SLIM_DET_DATASET_ROOT=C:\path\to\your\data

# Option 2: Pass dataset root
python quick_eval_dense.py --dataset_root C:\path\to\data --checkpoint last

# Option 3: Edit configs/data_quick.yaml with your paths
# Option 4: Pass explicit paths
python quick_eval_dense.py --val_images C:\data\images\val --val_labels C:\data\labels\val
```

### ❌ Model architecture mismatch
```bash
# Auto-loads from checkpoint, or match checkpoint config:
python quick_eval_dense.py \
    --checkpoint runs/dense_det/dense_det_best.pt \
    --backbone_name vst_custom \
    --variant small
```

---

## Key Files

| File | Purpose |
|------|---------|
| `quick_eval_dense.py` | Fast evaluation on 10-20 batches |
| `configs/dense_det_quick.yaml` | Fast training config (5 epochs) |
| `configs/dense_det.yaml` | Full training config (80 epochs) |
| `train_dense.py` | Full training loop |
| `evaluate_dense.py` | Complete evaluation |
| `model/dense_detector.py` | Model architecture |
| `model/dense_blocks.py` | Custom block implementations |

---

## Optimization Checklist

- [ ] Can each block train and eval in < 30 seconds on 10 batches?
- [ ] Does loss breakdown show reasonable component contributions?
- [ ] Does model support batch sizes 4, 8, 16, 32 without OOM?
- [ ] Can you iterate 5 times per minute?
- [ ] Are gradients flowing (no NaN, loss decreasing)?
- [ ] Does training full config converge?
