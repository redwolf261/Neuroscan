# Phase 4: Instrumentation & Diagnostics Guide

**Objective**: Add diagnostic logging to understand loss behavior, gradient flow, and parameter interactions without modifying the training algorithm.

**Strategy**: Add diagnostics **one step at a time**, testing after each step.

---

## Overview: 3-Step Instrumentation

```
Step 1: Individual Loss Logging
├─ Track: Dice, FocalTversky, Hybrid, Evidential, Total losses
├─ Output: loss_diagnostics.csv
├─ Overhead: Minimal (~0.1% slowdown)
└─ Reveals: Which losses are active, magnitude changes per epoch

Step 2: Gradient Norm Logging
├─ Track: Total gradient norm, component-wise (encoder/decoder/CBAM), per-loss
├─ Output: gradient_norms.csv, layer_wise_gradients.csv
├─ Overhead: Low (~2% slowdown)
└─ Reveals: Which components are being optimized, loss influence

Step 3: Gradient Similarity (OPTIONAL, computationally expensive)
├─ Track: Cosine similarity between gradients from different losses
├─ Output: gradient_similarity.csv
├─ Overhead: HIGH (~20-30% slowdown) - use selectively
└─ Reveals: If losses are aligned, independent, or conflicting
```

---

## Step 1: Individual Loss Logging

### What to do

1. Create `loss_logger` instance at start of `train_segmentation_epoch()`
2. After computing losses, call `log_batch()` with loss values
3. Call `flush_epoch()` at end of epoch
4. (Optional) Print epoch summary

### Code changes to `final_model.py`

**At the start of training loop** (around line 1829):

```python
from diagnostics.loss_logger import LossLogger, extract_loss_components, extract_evidential_loss

# Initialize logger (add once, before epoch loop)
loss_logger = LossLogger(SEGMENTATION_DIR, phase_name='segmentation')
```

**Inside `train_segmentation_epoch()` function** (around line 1367):

```python
# At the top of the function
loss_logger = LossLogger(SEGMENTATION_DIR, phase_name='segmentation')

# ... inside the batch loop (after computing losses)

with autocast(enabled=use_amp):
    out_dict = model(images)
    probs = out_dict["probs"]
    
    # Supervised loss
    loss_hybrid = criterion(probs, center_slice_label)
    
    # Extract individual components for logging
    hybrid_losses = extract_loss_components(criterion, probs, center_slice_label)
    
    # Evidential loss
    loss_evid = 0.0
    if USALD_ENABLED and ("alpha" in out_dict):
        loss_evid = extract_evidential_loss(evid_criterion, out_dict["alpha"], center_slice_label)
        loss_total = loss_hybrid + loss_evid
    else:
        loss_total = loss_hybrid

    # LOG LOSSES ← NEW
    learning_rates = {
        'encoder': optimizer.param_groups[0]['lr'],
        'decoder': optimizer.param_groups[1]['lr']
    }
    loss_logger.log_batch(
        epoch=epoch,
        batch_idx=batch_idx,
        losses_dict={
            'dice': hybrid_losses['dice'],
            'focal_tversky': hybrid_losses['focal_tversky'],
            'hybrid': loss_hybrid,
            'evidential': loss_evid,
            'total': loss_total
        },
        learning_rates=learning_rates
    )

# ... normal backward pass
optimizer.zero_grad()
scaler.scale(loss_total).backward()
scaler.step(optimizer)
scaler.update()

# ... at end of epoch loop
loss_logger.flush_epoch()

# Print epoch summary ← NEW
summary = loss_logger.get_epoch_summary(epoch)
if summary:
    print(f"\nLoss Summary Epoch {epoch}:")
    print(f"  Dice:      {summary['mean_dice']:.6f} ± {summary['std_dice']:.6f}")
    print(f"  FocalTver: {summary['mean_ft']:.6f} ± {summary['std_ft']:.6f}")
    print(f"  Hybrid:    {summary['mean_hybrid']:.6f} ± {summary['std_hybrid']:.6f}")
    print(f"  Evidential:{summary['mean_evid']:.6f} ± {summary['std_evid']:.6f}")
    print(f"  Total:     {summary['mean_total']:.6f} ± {summary['std_total']:.6f}")
```

### Testing Step 1

After making changes, run baseline test with 1 epoch:

```bash
cd 01_source_code/training_scripts
python resume_training.py --epochs 1 --batch_size 1
```

**Verify**:
- ✅ `segmentation_loss_diagnostics.csv` created in output directory
- ✅ CSV contains columns: epoch, batch, losses, learning_rates
- ✅ Training speed unchanged (minimal overhead)
- ✅ Loss values decrease from batch to batch

**Expected CSV output** (first few rows):
```
epoch,batch,timestamp,loss_dice,loss_focal_tversky,loss_hybrid,loss_evidential,loss_total,learning_rate_encoder,learning_rate_decoder
1,1,2026-07-27T10:00:00,0.999414,0.708804,0.854109,0.023451,0.877560,1.00e-05,4.00e-04
1,2,2026-07-27T10:00:05,0.998702,0.695321,0.847012,0.022134,0.869146,1.00e-05,4.00e-04
```

---

## Step 2: Gradient Norm Logging

### What to do

1. Create `gradient_logger` instance
2. After backward pass but before optimizer.step(), measure loss-specific gradients (OPTIONAL but valuable)
3. After optimizer.step(), measure total and component-wise gradient norms
4. Call `flush_batch()` at end of epoch

### Code changes to `final_model.py`

**At the start of training loop**:

```python
from diagnostics.gradient_logger import GradientLogger

# Initialize logger
grad_logger = GradientLogger(SEGMENTATION_DIR, seg_model, phase_name='segmentation')
```

**Inside batch loop** (around line 1418-1422, AFTER computing loss but BEFORE optimizer.step()):

```python
# Optional: Log loss-specific gradients (expensive, every Nth batch)
if batch_idx % 5 == 0:  # Every 5 batches, not every batch to reduce overhead
    grad_logger.log_loss_specific_gradients(
        epoch, batch_idx,
        loss_dice=...,  # Extract individual Dice loss from criterion
        loss_ft=...,    # Extract FT loss
        loss_evid=loss_evid,
        model=seg_model
    )
```

**After optimizer.step()** (around line 1422):

```python
# Log gradient norms ← NEW
grad_logger.log_batch_gradients(epoch, batch_idx, seg_model)

# Optional: Log layer-wise gradients (verbose, use sparingly)
if batch_idx % 10 == 0:  # Every 10 batches
    grad_logger.log_layer_wise_gradients(epoch, batch_idx, seg_model)
```

**At end of epoch**:

```python
grad_logger.flush_batch()
```

### Testing Step 2

```bash
python resume_training.py --epochs 1 --batch_size 1
```

**Verify**:
- ✅ `segmentation_gradient_norms.csv` created
- ✅ `segmentation_layer_wise_gradients.csv` created
- ✅ Contains: epoch, batch, timestamp, gradient norms by component
- ✅ Training speed ~2% slower (acceptable)

**Expected metrics** (interpret):
```
total_grad_norm,encoder_grad_norm,decoder_grad_norm,cbam_grad_norm
0.15234567,     0.08123456,       0.12456789,       0.03456789
```

**Analysis**:
- Encoder grad norm: Should increase if encoder is being fine-tuned
- Decoder grad norm: Typically larger (training from scratch)
- CBAM grad norm: Medium (specialized component)
- Ratio encoder/decoder: Should be ~1:10 to 1:50 (reflects LR difference)

---

## Step 3: Gradient Similarity (OPTIONAL)

### What to do (optional, for advanced analysis)

1. Create `gradient_similarity_logger`
2. Every Nth batch, compute gradient vectors for each loss separately
3. Compute cosine similarities between them
4. Analyze if losses are aligned or conflicting

### Code changes

```python
from diagnostics.gradient_similarity import GradientSimilarityLogger

sim_logger = GradientSimilarityLogger(SEGMENTATION_DIR, 'segmentation')

# Inside batch loop (around line 1418, BEFORE backward pass)
# Only every Nth batch due to computational cost
if batch_idx % 20 == 0:  # Every 20 batches
    sims = sim_logger.log_gradient_similarity(
        epoch, batch_idx, seg_model,
        loss_dice=...,      # Extract from criterion
        loss_ft=...,
        loss_evid=loss_evid
    )
    print(sim_logger.analyze_batch(sims))

# At end of epoch
sim_logger.flush()
```

**WARNING**: This is slow (~20-30% overhead). Use only:
- Early epochs to diagnose problems
- Every Nth batch (20+) to minimize impact
- Only when investigating specific loss conflicts

### Testing Step 3

```bash
python resume_training.py --epochs 1 --batch_size 1
```

**Expected output**:
```
✓ Dice & FocalTversky: ALIGNED (complementary)
✓ HybridLoss & Evidential: Independent (OK)
✓ Overall: NO CONFLICT detected
```

**Interpretation**:
- Cosine similarity > 0.9: Losses perfectly aligned
- Cosine similarity 0.1-0.8: Independent (OK)
- Cosine similarity < 0: Conflicting (problematic)

---

## Full Integration Example

Here's how a modified `train_segmentation_epoch()` looks with all diagnostics:

```python
def train_segmentation_epoch_with_diagnostics(model, loader, criterion, optimizer, scaler, 
                                             device, teacher_model=None, epoch=1, tau_pl=0.5):
    """Training with diagnostics."""
    model.train()
    if teacher_model is not None:
        teacher_model.eval()

    # Initialize loggers
    loss_logger = LossLogger(SEGMENTATION_DIR, 'segmentation')
    grad_logger = GradientLogger(SEGMENTATION_DIR, model, 'segmentation')
    sim_logger = GradientSimilarityLogger(SEGMENTATION_DIR, 'segmentation')

    total_loss = 0
    total_dice = 0

    evid_criterion = EvidentialBetaLoss(lambda_kl=LAMBDA_EVIDENTIAL) if USALD_ENABLED else None
    use_consistency = USALD_CONSISTENCY_ENABLED and teacher_model is not None and epoch > WARMUP_EPOCHS

    pbar = tqdm(loader, desc="Segmentation Training")
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        center_slice_label = labels[:, :, labels.shape[2]//2, :, :]

        with autocast(enabled=use_amp):
            out_dict = model(images)
            probs = out_dict["probs"]

            # Compute losses
            loss_hybrid = criterion(probs, center_slice_label)
            hybrid_losses = extract_loss_components(criterion, probs, center_slice_label)

            loss_evid = 0.0
            if USALD_ENABLED and ("alpha" in out_dict):
                loss_evid = extract_evidential_loss(evid_criterion, out_dict["alpha"], center_slice_label)
                loss_total = loss_hybrid + loss_evid
            else:
                loss_total = loss_hybrid

            # LOG LOSSES
            loss_logger.log_batch(
                epoch, batch_idx,
                losses_dict={
                    'dice': hybrid_losses['dice'],
                    'focal_tversky': hybrid_losses['focal_tversky'],
                    'hybrid': loss_hybrid,
                    'evidential': loss_evid,
                    'total': loss_total
                },
                learning_rates={
                    'encoder': optimizer.param_groups[0]['lr'],
                    'decoder': optimizer.param_groups[1]['lr']
                }
            )

        # Optional: Log similarity (expensive)
        if batch_idx % 20 == 0 and batch_idx > 0:
            sims = sim_logger.log_gradient_similarity(epoch, batch_idx, model, loss_hybrid, ..., loss_evid)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss_total).backward()
        
        # LOG GRADIENTS (after backward, before step)
        if batch_idx % 5 == 0:
            grad_logger.log_loss_specific_gradients(epoch, batch_idx, loss_hybrid, ..., loss_evid, model)

        scaler.step(optimizer)
        scaler.update()

        # LOG GRADIENTS (after step)
        grad_logger.log_batch_gradients(epoch, batch_idx, model)
        if batch_idx % 10 == 0:
            grad_logger.log_layer_wise_gradients(epoch, batch_idx, model)

        # Compute metrics
        pred_binary = (probs > 0.5).float()
        dice = 2 * (pred_binary * center_slice_label).sum() / (pred_binary.sum() + center_slice_label.sum() + 1e-7)

        total_loss += loss_total.item()
        total_dice += dice.item()

        pbar.set_postfix({"loss": loss_total.item(), "dice": dice.item()})

    # Flush all logs
    loss_logger.flush_epoch()
    grad_logger.flush_batch()
    sim_logger.flush()

    # Print epoch summary
    summary = loss_logger.get_epoch_summary(epoch)
    if summary:
        print(f"\n=== Epoch {epoch} Diagnostics ===")
        print(f"Losses:")
        print(f"  Dice:      {summary['mean_dice']:.6f} ± {summary['std_dice']:.6f}")
        print(f"  FocalTver: {summary['mean_ft']:.6f} ± {summary['std_ft']:.6f}")
        print(f"  Evidential:{summary['mean_evid']:.6f} ± {summary['std_evid']:.6f}")
        print(f"  Total:     {summary['mean_total']:.6f} ± {summary['std_total']:.6f}")

    return total_loss / len(loader), total_dice / len(loader)
```

---

## What to Look For

### After Step 1 (Loss Logging)

**Q: Do losses decrease?**
- ✅ Good: Dice 0.99→0.95→0.90 over 10 batches
- ❌ Bad: Dice stays flat or increases
- ⚠️ Oscillating: Large batch-to-batch fluctuations

**Q: Are all losses active?**
- ✅ Good: Both Hybrid and Evidential > 0
- ⚠️ Watch: Evidential consistently ~1e-4 (very small, good)
- ❌ Bad: Evidential = 0 (not computing)

### After Step 2 (Gradient Logging)

**Q: Are encoder/decoder both updating?**
- ✅ Good: Encoder grad norm > 0.01, Decoder > 0.05
- ⚠️ Watch: Encoder grad norm very small (fine-tuning, expected)
- ❌ Bad: Encoder grad norm = 0 (frozen when shouldn't be)

**Q: What's the gradient magnitude progression?**
- ✅ Good: Stable or slightly decreasing per epoch
- ⚠️ Watch: Exploding (grad norm > 1.0, use gradient clipping)
- ❌ Bad: Vanishing (grad norm < 1e-8, learning rate too small)

### After Step 3 (Similarity Logging)

**Q: Are losses aligned?**
- ✅ Good: Cosine sim (Dice vs FT) > 0.7 (aligned)
- ✅ Good: Cosine sim (Hybrid vs Evid) > 0.3 (independent, OK)
- ⚠️ Watch: Cosine sim ~ 0 (independent, might need investigation)
- ❌ Bad: Cosine sim < 0 (conflicting, losses fighting)

---

## Performance Impact

| Step | Overhead | Frequency | When to use |
|------|----------|-----------|------------|
| 1 (Loss) | 0.1% | Every batch | Always (minimal cost) |
| 2 (Gradients) | 2% | Every batch | Always after tuning |
| 3 (Similarity) | 20-30% | Every 20th batch | Diagnostic/debug only |

---

## Implementation Checklist

- [ ] Create `diagnostics/` directory with 3 logger modules
- [ ] Step 1: Add loss logging to `train_segmentation_epoch()`
- [ ] Test Step 1: Run 1-epoch training, verify CSV created
- [ ] Step 2: Add gradient logging
- [ ] Test Step 2: Run 1-epoch training, verify gradient norms reasonable
- [ ] Step 3 (optional): Add similarity logging for problem investigation
- [ ] Run full training with diagnostics, collect data
- [ ] Analyze CSV outputs for patterns and insights

---

## Next: Analysis Phase

Once diagnostics are running, you'll have CSVs to analyze. Questions to investigate:

1. **Loss domination**: Plot each loss component over epochs. Which dominates?
2. **Gradient flow**: Are encoder/decoder ratio matching LR difference?
3. **Convergence**: Do losses plateau? At what point?
4. **Stability**: Are metrics (loss, dice) stable or noisy?
5. **Conflicting gradients**: Any negative cosine similarities?

**Ready to implement? Start with Step 1 only.**
