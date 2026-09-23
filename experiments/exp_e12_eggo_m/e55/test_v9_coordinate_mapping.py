"""
Phase E55, verification check #1: coordinate-mapping unit test.

Verifies the full pipeline -- 64^3-grid coordinate -> native voxel
coordinate (per-axis scalar ratio) -> grid_sample normalized [-1,1]
coordinate (align_corners=True) -- correctly recovers a known synthetic
marker placed at a known native location. This is the concrete,
numeric test for the half-voxel-alignment bug risk identified during
design (scipy.ndimage.zoom's own indexing convention does not match
grid_sample's align_corners semantics).

Must pass before any other verification step or real training.
"""
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import zoom


def native_to_norm(coord, size):
    """align_corners=True convention: index 0 -> -1, index (size-1) -> +1."""
    return 2.0 * coord / (size - 1) - 1.0


def resized_to_native(coord_resized, current_shape, target_shape, axis):
    """Per-axis scalar-ratio mapping, matching scipy.ndimage.zoom's own
    convention (zoom_factor = target/current; i_in = i_out / zoom_factor
    = i_out * current/target)."""
    return coord_resized * (current_shape[axis] / target_shape[axis])


def extract_crop(native_volume, center_native, crop_size, native_shape):
    """native_volume: (1,1,D,H,W) tensor. center_native: (3,) tensor,
    native voxel coords (D,H,W order). Returns (1,1,crop,crop,crop)."""
    offsets = torch.arange(crop_size).float() - (crop_size - 1) / 2.0
    grid_native = center_native.view(3, 1, 1, 1) + torch.stack(
        torch.meshgrid(offsets, offsets, offsets, indexing="ij"), dim=0
    )  # (3, crop, crop, crop)

    grid_norm = torch.stack([
        native_to_norm(grid_native[i], native_shape[i]) for i in range(3)
    ], dim=-1)  # (crop, crop, crop, 3), order (D,H,W)

    # grid_sample expects grid channels in (W,H,D) order for a 5D input
    # (matching x,y,z convention), NOT (D,H,W) -- this is a second,
    # separate potential source of a coordinate-order bug, distinct from
    # the align_corners issue, and is explicitly tested for below.
    grid_norm_xyz = grid_norm[..., [2, 1, 0]].unsqueeze(0)  # (1, crop,crop,crop, 3), (W,H,D)->(x,y,z)

    crop = F.grid_sample(
        native_volume, grid_norm_xyz, mode="nearest",
        padding_mode="zeros", align_corners=True,
    )
    return crop


def main():
    native_shape = (240, 240, 155)
    target_shape = (64, 64, 64)

    # Place a synthetic marker at a KNOWN native location. Use a small
    # BLOB (not a single voxel) for Test 1 specifically -- a single
    # isolated voxel at this downsampling ratio (240/64 ~= 3.75x/axis)
    # is exactly the pathological case E29 already documented (median
    # native lesion component vanishes entirely under nearest-neighbor
    # resize); that's a real, already-known project finding, not a bug
    # in the coordinate-mapping formula being tested here. A ~9-voxel
    # cube blob survives resize reliably and isolates the mapping
    # question this test is actually meant to check.
    marker_native = (120, 100, 80)  # (D,H,W), well inside bounds, blob center
    native_volume_np = np.zeros(native_shape, dtype=np.float32)
    native_volume_np[marker_native[0]-1:marker_native[0]+2,
                      marker_native[1]-1:marker_native[1]+2,
                      marker_native[2]-1:marker_native[2]+2] = 1.0

    # ================= Test 1: resized-grid -> native mapping correctness =================
    # Resize the SAME volume down to 64^3 (nearest, matching mask convention)
    # and find where the marker landed in the resized grid.
    zoom_factors = tuple(t / c for t, c in zip(target_shape, native_shape))
    resized_np = zoom(native_volume_np, zoom_factors, order=0)
    if resized_np.max() == 0:
        raise RuntimeError("Test 1 setup failed: blob marker still vanished after resize -- "
                            "increase blob size before trusting this test's result.")
    # Blob resizes to a small cluster of nonzero voxels, not a single
    # point -- use the CENTROID of the resized cluster (probability-
    # weighted mean, matching the real mechanism's own soft-centroid
    # approach) rather than argmax, for a fair, representative check.
    nz = np.argwhere(resized_np > 0)
    resized_marker_idx = tuple(nz.mean(axis=0))
    print(f"Native marker location: {marker_native}")
    print(f"Marker location after resize to 64^3: {resized_marker_idx}")

    # Map the RESIZED marker location back to native coords via our formula.
    recovered_native = tuple(
        resized_to_native(resized_marker_idx[i], native_shape, target_shape, i)
        for i in range(3)
    )
    print(f"Recovered native coords (via scalar-ratio mapping): {recovered_native}")
    max_err = max(abs(recovered_native[i] - marker_native[i]) for i in range(3))
    print(f"Max coordinate error: {max_err:.3f} voxels")
    # Tolerance: resize with order=0 (nearest) can shift by up to ~1 voxel
    # due to quantization at 64^3 resolution (240/64 ~= 3.75x downsample
    # per axis) -- a few-voxel tolerance is expected and NOT a bug; this
    # check is for a GROSS misalignment (e.g. axis swap, off-by-many, or
    # a systematic offset), not sub-voxel precision.
    test1_pass = max_err < 5.0
    print(f"TEST 1 (resized->native mapping, coarse sanity) PASS: {test1_pass}\n")

    # ================= Test 2: grid_sample crop correctly centers on a KNOWN native coord =================
    # This is the precise test: center a crop EXACTLY on a SINGLE-VOXEL
    # marker (separate volume from Test 1's blob) and verify grid_sample
    # recovers the marker at the crop's own center voxel. Skips the
    # resize-quantization step entirely -- tests the grid_sample mapping
    # in isolation.
    native_volume_np_single = np.zeros(native_shape, dtype=np.float32)
    native_volume_np_single[marker_native] = 1.0
    native_volume_t = torch.from_numpy(native_volume_np_single).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    center_native_t = torch.tensor(marker_native, dtype=torch.float32)
    crop_size = 33  # odd, so there's an exact center voxel

    crop = extract_crop(native_volume_t, center_native_t, crop_size, native_shape)
    crop_center_idx = crop_size // 2
    marker_value_at_center = crop[0, 0, crop_center_idx, crop_center_idx, crop_center_idx].item()
    total_marker_mass = crop.sum().item()
    print(f"Crop shape: {crop.shape}")
    print(f"Value at crop's own center voxel: {marker_value_at_center:.4f} (expect ~1.0)")
    print(f"Total marker mass recovered in crop: {total_marker_mass:.4f} (expect ~1.0, marker not lost/duplicated)")
    test2_pass = marker_value_at_center > 0.99 and abs(total_marker_mass - 1.0) < 1e-3
    print(f"TEST 2 (grid_sample crop centering, precise) PASS: {test2_pass}\n")

    # ================= Test 3: off-center marker within the crop is at the right relative position =================
    marker2_native = (125, 105, 85)  # 5 voxels off from crop center in each axis
    native_volume_np2 = np.zeros(native_shape, dtype=np.float32)
    native_volume_np2[marker2_native] = 1.0
    native_volume_t2 = torch.from_numpy(native_volume_np2).unsqueeze(0).unsqueeze(0)
    crop2 = extract_crop(native_volume_t2, center_native_t, crop_size, native_shape)  # crop centered on marker_native, NOT marker2
    found_idx = np.unravel_index(torch.argmax(crop2).item(), crop2.shape[2:])
    expected_offset = tuple(marker2_native[i] - marker_native[i] for i in range(3))
    expected_idx = tuple(crop_center_idx + expected_offset[i] for i in range(3))
    print(f"Off-center marker found at crop-local index: {found_idx}, expected: {expected_idx}")
    test3_pass = found_idx == expected_idx
    print(f"TEST 3 (off-center relative positioning, catches axis-order bugs) PASS: {test3_pass}\n")

    all_pass = test1_pass and test2_pass and test3_pass
    print(f"{'='*60}\nALL COORDINATE-MAPPING TESTS PASS: {all_pass}")
    if not all_pass:
        print("!!! DO NOT PROCEED TO NEXT VERIFICATION STEP UNTIL ALL TESTS PASS !!!")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
