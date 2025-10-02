import open3d as o3d
import laspy
import numpy as np
import argparse
import sys

def convert_pcd_to_las_with_color(pcd_path, las_path):
    """
    Converts a PCD file with color to a LAS file with color.

    Args:
        pcd_path (str): Path to the input PCD file.
        las_path (str): Path to the output LAS file.
    """
    # 1. Read the PCD file using Open3D
    pcd = o3d.io.read_point_cloud(pcd_path)

    if not pcd.has_points():
        print(f"❌ The PCD file {pcd_path} is empty.")
        sys.exit(1)

    # 2. Create a new LAS file
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)

    # 3. Set the coordinates
    pts = np.asarray(pcd.points)
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]

    # 4. Set the color information
    if pcd.has_colors():
        # Open3D colors are in [0, 1]; LAS expects [0, 65535]
        colors = (np.asarray(pcd.colors) * 65535).clip(0, 65535)
        las.red = colors[:, 0].astype(np.uint16)
        las.green = colors[:, 1].astype(np.uint16)
        las.blue = colors[:, 2].astype(np.uint16)
    else:
        print("⚠️  No color information found in PCD — LAS will be written without colors.")

    # 5. Write the LAS file
    las.write(las_path)
    print(f"✅ Successfully converted {pcd_path} → {las_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert PCD to LAS with color using Open3D + laspy.")
    parser.add_argument("input_pcd", help="Path to the input PCD file")
    parser.add_argument("output_las", help="Path to the output LAS file")
    args = parser.parse_args()

    convert_pcd_to_las_with_color(args.input_pcd, args.output_las)


if __name__ == "__main__":
    main()
