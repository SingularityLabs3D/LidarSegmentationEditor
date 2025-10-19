#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import laspy
import open3d as o3d

def _to_uint16_color(colors_f):
    """
    Accepts Nx3 colors as floats. If max <= 1.001 -> assume 0..1, else assume 0..255.
    Returns uint16 Nx3 in 0..65535.
    """
    if colors_f.size == 0:
        return colors_f.astype(np.uint16)

    finite = np.isfinite(colors_f).all(axis=1)
    # Treat non-finite as missing
    colors_f = colors_f.copy()
    colors_f[~finite] = 1.0

    mx = float(np.nanmax(colors_f)) if colors_f.size else 1.0
    if mx <= 1.001:
        # 0..1 range
        scaled = (colors_f * 65535.0).clip(0, 65535).astype(np.uint16)
    else:
        # 0..255 range
        scaled = (colors_f.clip(0, 255) * 257.0).astype(np.uint16)
    return scaled

def _ensure_colors(pcd):
    """
    Returns Nx3 float colors where all invalid/missing are replaced with white.
    White = [1,1,1] in 0..1 space; we detect scale later.
    """
    n = np.asarray(pcd.points).shape[0]
    if n == 0:
        return np.empty((0, 3), dtype=np.float32)

    if pcd.has_colors():
        col = np.asarray(pcd.colors)  # Open3D gives float64 in 0..1 usually
        # Fix shape mismatches
        if col.shape[0] != n or col.shape[1] != 3:
            col = np.ones((n, 3), dtype=np.float32)  # all white
        else:
            col = col.astype(np.float32, copy=True)
            # Missing if any component is non-finite, or all three are ~0
            nonfinite = ~np.isfinite(col).all(axis=1)
            near_zero = (np.abs(col).sum(axis=1) < 1e-12)
            missing_mask = nonfinite | near_zero
            if missing_mask.any():
                col[missing_mask] = 1.0  # white
    else:
        col = np.ones((n, 3), dtype=np.float32)  # all white

    return col

def convert_pcd_to_las_with_color(pcd_path, las_path):
    """
    Convert PCD -> LAS, preserving existing colors and setting white for points with no color.
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd.has_points():
        print(f"ERROR: The PCD file {pcd_path} has no points.", file=sys.stderr)
        sys.exit(1)

    pts = np.asarray(pcd.points)
    colors_f = _ensure_colors(pcd)              # Nx3 float, invalid -> white
    colors16 = _to_uint16_color(colors_f)       # Nx3 uint16 0..65535

    # Create LAS (use point format 2: XYZ + RGB)
    header = laspy.LasHeader(point_format=2, version="1.2")
    las = laspy.LasData(header)

    las.x = pts[:, 0].astype(np.float64)
    las.y = pts[:, 1].astype(np.float64)
    las.z = pts[:, 2].astype(np.float64)

    las.red   = colors16[:, 0]
    las.green = colors16[:, 1]
    las.blue  = colors16[:, 2]

    las.write(las_path)
    print(f"Converted {pcd_path} -> {las_path} with colors (white filled where missing).")

def main():
    ap = argparse.ArgumentParser(description="Convert PCD to LAS, assigning white to points without color.")
    ap.add_argument("input_pcd", help="Path to input PCD (uncompressed binary or ASCII; Open3D will read it).")
    ap.add_argument("output_las", help="Path to output LAS.")
    args = ap.parse_args()
    convert_pcd_to_las_with_color(args.input_pcd, args.output_las)

if __name__ == "__main__":
    main()

