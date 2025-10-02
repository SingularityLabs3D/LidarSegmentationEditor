import subprocess
import time
import shlex
import os

def run_step(cmd, desc):
    print(f"\n=== {desc} ===")
    print("Running:", cmd)
    start = time.perf_counter()
    try:
        # run and capture output
        result = subprocess.run(shlex.split(cmd), check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {desc} failed with return code {e.returncode}")
        print(f"STDERR: {e.stderr}")
        raise
    end = time.perf_counter()
    elapsed = end - start
    print(f"[DONE] {desc} took {elapsed:.2f} seconds")
    return elapsed

def run_conversion_pipeline(input_pcd_path: str, output_dir: str):
    total_start = time.perf_counter()
    
    os.makedirs(output_dir, exist_ok=True)
    
    uncompressed_pcd = os.path.join(output_dir, "points_uncompressed.pcd")
    colored_las = os.path.join(output_dir, "points_colored.las")
    potree_output = os.path.join(output_dir, "potree_output")

    steps = [
        (
            f"pcl_convert_pcd_ascii_binary {input_pcd_path} {uncompressed_pcd} 1",
            "Uncompress PCD → Binary"
        ),
        (
            f"python diy_pdal.py {uncompressed_pcd} {colored_las}",
            "Convert Binary PCD → LAS"
        ),
        (
            f"./PotreeConverter {colored_las} -o {potree_output}",
            "Convert LAS → Potree format"
        )
    ]

    timings = {}
    for cmd, desc in steps:
        elapsed = run_step(cmd, desc)
        timings[desc] = elapsed

    total_end = time.perf_counter()
    timings["Total pipeline"] = total_end - total_start

    print("\n=== Summary ===")
    for desc, t in timings.items():
        print(f"{desc}: {t:.2f} s")
    
    return timings

def main():
    input_pcd_path = "first_chunk_22.pcd"
    output_dir = "./"
    run_conversion_pipeline(input_pcd_path, output_dir)

if __name__ == "__main__":
    main()
