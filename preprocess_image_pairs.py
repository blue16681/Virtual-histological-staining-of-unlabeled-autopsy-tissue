import argparse
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def normalize_key(path):
    key = path.stem.lower()
    key = re.sub(r"\bunstained\b", "", key)
    key = re.sub(r"\s+", " ", key)
    return key.replace(" ", "")


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def pair_images(input_dir, target_dir):
    inputs = {normalize_key(p): p for p in list_images(input_dir)}
    targets = {normalize_key(p): p for p in list_images(target_dir)}
    keys = sorted(set(inputs) & set(targets))
    return [(inputs[k], targets[k], k) for k in keys]


def tissue_fraction(patch, white_threshold):
    return float(np.mean(np.any(patch < white_threshold, axis=-1)))


def estimate_rigid_transform(moving_bgr, fixed_bgr, max_width=1600, iterations=300):
    fixed_gray = cv2.cvtColor(fixed_bgr, cv2.COLOR_BGR2GRAY)
    moving_gray = cv2.cvtColor(moving_bgr, cv2.COLOR_BGR2GRAY)
    h, w = fixed_gray.shape[:2]
    scale = min(1.0, max_width / max(h, w))

    if scale < 1.0:
        fixed_small = cv2.resize(fixed_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        moving_small = cv2.resize(moving_gray, (fixed_small.shape[1], fixed_small.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        fixed_small = fixed_gray
        moving_small = cv2.resize(moving_gray, (w, h), interpolation=cv2.INTER_AREA)

    fixed_small = cv2.equalizeHist(fixed_small).astype(np.float32) / 255.0
    moving_small = cv2.equalizeHist(moving_small).astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1e-6)

    try:
        cc, warp_small = cv2.findTransformECC(
            fixed_small, moving_small, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5
        )
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32), 0.0, False

    warp_full = warp_small.copy()
    if scale < 1.0:
        warp_full[:, 2] /= scale
    return warp_full.astype(np.float32), float(cc), True


def estimate_local_transform(moving_bgr, fixed_bgr, mode="translation", max_shift=32, iterations=80):
    if mode == "none":
        return np.eye(2, 3, dtype=np.float32), 0.0, True

    motion_type = {
        "translation": cv2.MOTION_TRANSLATION,
        "rigid": cv2.MOTION_EUCLIDEAN,
    }[mode]
    fixed_gray = cv2.cvtColor(fixed_bgr, cv2.COLOR_BGR2GRAY)
    moving_gray = cv2.cvtColor(moving_bgr, cv2.COLOR_BGR2GRAY)
    fixed_gray = cv2.equalizeHist(fixed_gray).astype(np.float32) / 255.0
    moving_gray = cv2.equalizeHist(moving_gray).astype(np.float32) / 255.0

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1e-5)
    try:
        cc, warp = cv2.findTransformECC(
            fixed_gray, moving_gray, warp, motion_type, criteria, None, 3
        )
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32), 0.0, False

    shift = np.linalg.norm(warp[:, 2])
    if shift > max_shift:
        return np.eye(2, 3, dtype=np.float32), float(cc), False
    return warp.astype(np.float32), float(cc), True


def refine_patch(input_patch, target_patch, local_registration, max_shift):
    if local_registration == "none":
        return input_patch, False, 0.0

    warp, ecc, ok = estimate_local_transform(
        input_patch, target_patch, mode=local_registration, max_shift=max_shift
    )
    if not ok:
        return input_patch, False, ecc

    refined = cv2.warpAffine(
        input_patch,
        warp,
        (target_patch.shape[1], target_patch.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return refined, True, ecc


def save_pair_patches(input_img, target_img, sample_key, split, out_root, patch_size, white_threshold, min_tissue,
                      local_registration, local_max_shift):
    out_input = out_root / split / "input"
    out_target = out_root / split / "target"
    out_input.mkdir(parents=True, exist_ok=True)
    out_target.mkdir(parents=True, exist_ok=True)

    h, w = target_img.shape[:2]
    saved = 0
    skipped_white = 0
    local_ok = 0
    local_failed = 0
    local_ecc_values = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            inp_patch = input_img[y:y + patch_size, x:x + patch_size]
            tgt_patch = target_img[y:y + patch_size, x:x + patch_size]
            if (tissue_fraction(inp_patch, white_threshold) < min_tissue or
                    tissue_fraction(tgt_patch, white_threshold) < min_tissue):
                skipped_white += 1
                continue

            inp_patch, ok, local_ecc = refine_patch(inp_patch, tgt_patch, local_registration, local_max_shift)
            if local_registration != "none":
                local_ecc_values.append(local_ecc)
                if ok:
                    local_ok += 1
                else:
                    local_failed += 1

            name = f"{sample_key}_x{x:05d}_y{y:05d}.png"
            cv2.imwrite(str(out_input / name), inp_patch)
            cv2.imwrite(str(out_target / name), tgt_patch)
            saved += 1

    local_ecc_mean = float(np.mean(local_ecc_values)) if local_ecc_values else 0.0
    return saved, skipped_white, local_ok, local_failed, local_ecc_mean


def main():
    parser = argparse.ArgumentParser(
        description="Register image pairs, tile 256x256 patches, remove white patches, and split train/test."
    )
    parser.add_argument("--input-dir", default="dataset/DermaRepo/IHC")
    parser.add_argument("--target-dir", default="dataset/DermaRepo/HE")
    parser.add_argument("--output-dir", default="dataset/DermaRepo_processed")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--white-threshold", type=int, default=245)
    parser.add_argument("--min-tissue", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-registration-width", type=int, default=1600)
    parser.add_argument("--local-registration", choices=["none", "translation", "rigid"], default="translation",
                        help="Patch-level refinement after whole-image rigid registration.")
    parser.add_argument("--local-max-shift", type=float, default=32,
                        help="Reject patch-level refinement if the predicted shift is larger than this many pixels.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    target_dir = Path(args.target_dir)
    out_root = Path(args.output_dir)
    pairs = pair_images(input_dir, target_dir)
    if not pairs:
        raise RuntimeError(f"No paired images found between {input_dir} and {target_dir}")

    random.Random(args.seed).shuffle(pairs)
    n_test = max(1, int(round(len(pairs) * args.test_ratio)))
    test_keys = {key for _, _, key in pairs[:n_test]}

    manifest = {
        "input_dir": str(input_dir),
        "target_dir": str(target_dir),
        "output_dir": str(out_root),
        "patch_size": args.patch_size,
        "test_ratio": args.test_ratio,
        "white_threshold": args.white_threshold,
        "min_tissue": args.min_tissue,
        "local_registration": args.local_registration,
        "local_max_shift": args.local_max_shift,
        "pairs": [],
    }

    for idx, (input_path, target_path, key) in enumerate(pairs, 1):
        split = "test" if key in test_keys else "train"
        print(f"[{idx}/{len(pairs)}] {split}: {input_path.name} -> {target_path.name}")
        moving = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        fixed = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
        if moving is None or fixed is None:
            print("  skipped: failed to read image")
            continue

        fixed_h, fixed_w = fixed.shape[:2]
        if moving.shape[:2] != fixed.shape[:2]:
            moving = cv2.resize(moving, (fixed_w, fixed_h), interpolation=cv2.INTER_AREA)

        warp, ecc, ok = estimate_rigid_transform(moving, fixed, max_width=args.max_registration_width)
        aligned = cv2.warpAffine(
            moving, warp, (fixed_w, fixed_h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        saved, skipped_white, local_ok, local_failed, local_ecc_mean = save_pair_patches(
            aligned, fixed, key, split, out_root, args.patch_size,
            args.white_threshold, args.min_tissue,
            args.local_registration, args.local_max_shift,
        )
        print(
            f"  registration_ok={ok} ecc={ecc:.4f} saved={saved} skipped_white={skipped_white} "
            f"local_ok={local_ok} local_failed={local_failed} local_ecc_mean={local_ecc_mean:.4f}"
        )
        manifest["pairs"].append({
            "key": key,
            "split": split,
            "input": str(input_path),
            "target": str(target_path),
            "registration_ok": ok,
            "ecc": ecc,
            "saved_patches": saved,
            "skipped_white_patches": skipped_white,
            "local_registration": args.local_registration,
            "local_ok_patches": local_ok,
            "local_failed_patches": local_failed,
            "local_ecc_mean": local_ecc_mean,
            "warp_input_to_target": warp.tolist(),
        })

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
