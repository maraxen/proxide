import gc
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import psutil
from tqdm import tqdm

import proxide


def get_process_memory():
    """Get current process memory (RSS) in MiB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def create_synthetic_trajectory(filename, n_atoms, n_frames):
    """Create a large synthetic DCD trajectory."""
    print(f"Generating synthetic DCD: {n_atoms} atoms, {n_frames} frames...")

    # Pre-calculate a single frame
    coords = np.random.rand(n_atoms, 3).astype(np.float32)

    with proxide.write_dcd(filename, n_atoms, delta=1.0) as writer:
        for _ in tqdm(range(n_frames), desc="Writing DCD"):
            writer.write_frame(coords)

def run_bench():
    # Configuration
    N_ATOMS = 10000
    # Scaling test: 100 to 10000 frames
    FRAME_SIZES = [100, 500, 1000, 5000, 10000]
    CHUNK_SIZE = 100
    TRAJ_PATH = "tmp/bench_scaling.dcd"

    os.makedirs("tmp", exist_ok=True)
    os.makedirs("benchmarks/results", exist_ok=True)

    results = []

    print("=" * 60)
    print("PROXIDE STREAMING PERFORMANCE BENCHMARK")
    print("=" * 60)

    for n_frames in FRAME_SIZES:
        # 1. Setup
        create_synthetic_trajectory(TRAJ_PATH, N_ATOMS, n_frames)
        file_size_gb = os.path.getsize(TRAJ_PATH) / (1024**3)
        print(f"File size: {file_size_gb:.2f} GB")

        # 2. Benchmark Proxide Streaming
        gc.collect()
        initial_mem = get_process_memory()

        start_time = time.perf_counter()
        peak_mem = initial_mem

        count = 0
        for chunk in proxide.iterload(TRAJ_PATH, chunk_size=CHUNK_SIZE):
            count += chunk["coordinates"].shape[0]
            current_mem = get_process_memory()
            peak_mem = max(peak_mem, current_mem)

        end_time = time.perf_counter()
        duration = end_time - start_time
        throughput = count / duration if duration > 0 else 0

        mem_increase = peak_mem - initial_mem

        print(f"Result for {n_frames} frames:")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Throughput: {throughput:.0f} frames/s")
        print(f"  Peak Memory Increase: {mem_increase:.2f} MiB")

        results.append({
            "n_frames": n_frames,
            "duration": duration,
            "throughput": throughput,
            "mem_increase": mem_increase,
            "file_size": file_size_gb
        })

        # Cleanup file to save space
        if os.path.exists(TRAJ_PATH):
            os.remove(TRAJ_PATH)

    # 3. Visualization
    n_frames_arr = [r["n_frames"] for r in results]
    mem_inc_arr = [r["mem_increase"] for r in results]
    throughput_arr = [r["throughput"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Memory Scale Plot
    ax1.plot(n_frames_arr, mem_inc_arr, 'o-', color='#e74c3c', linewidth=2, markersize=8)
    ax1.set_title("Memory Stability (O(1) Check)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Number of Frames", fontsize=12)
    ax1.set_ylabel("Peak Memory Increase (MiB)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(0, max(mem_inc_arr) * 1.5 if mem_inc_arr else 100)

    # Throughput Plot
    ax2.plot(n_frames_arr, throughput_arr, 'o-', color='#3498db', linewidth=2, markersize=8)
    ax2.set_title("Streaming Throughput", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Number of Frames", fontsize=12)
    ax2.set_ylabel("Frames per Second", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = "benchmarks/results/memory_profile.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nBenchmark report saved to: {plot_path}")

    # Summary of O(1)
    slope = np.polyfit(n_frames_arr, mem_inc_arr, 1)[0]
    print("\n" + "="*60)
    print(f"O(1) ANALYSIS: Memory growth slope = {slope:.6f} MiB/frame")
    if slope < 0.001:
        print("VERDICT: PASS (O(1) memory usage confirmed)")
    else:
        print("VERDICT: WARNING (Memory growth detected)")
    print("="*60)

if __name__ == "__main__":
    run_bench()
