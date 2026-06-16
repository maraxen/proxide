import numpy as np
import json
import math
import sys
import os

def kabsch_alignment(source_pts, target_pts):
    centroid_src = np.mean(source_pts, axis=0)
    centroid_tgt = np.mean(target_pts, axis=0)
    centered_src = source_pts - centroid_src
    centered_tgt = target_pts - centroid_tgt
    
    h = np.dot(centered_src.T, centered_tgt)
    u, s, vh = np.linalg.svd(h)
    
    rot = np.dot(u, vh)
    if np.linalg.det(rot) < 0:
        rot = np.dot(u, np.dot(np.diag([1, 1, -1]), vh))
        
    return rot, centroid_src, centroid_tgt

def benchmark():
    template_atoms = {
        "N": [0.0, 0.0, 0.0],
        "CA": [1.0, 0.0, 0.0],
        "CB": [1.0, 1.0, 0.0],
        "C": [1.0, 1.0, 1.0],
        "O": [2.0, 0.0, 1.0]
    }
    
    existing_atoms = {
        "N": [10.0, 10.0, 10.0],
        "CA": [11.0, 10.0, 10.0],
        "C": [11.0, 11.0, 11.0]
    }
    
    common = sorted(list(set(template_atoms.keys()) & set(existing_atoms.keys())))
    source_pts = np.array([existing_atoms[a] for a in common])
    target_pts = np.array([template_atoms[a] for a in common])
    
    rot, centroid_src, centroid_tgt = kabsch_alignment(source_pts, target_pts)
    
    reconstructed = {}
    for name, pos in template_atoms.items():
        if name not in existing_atoms:
            centered_ta = np.array(pos) - centroid_tgt
            rotated = np.dot(centered_ta, rot.T)
            final_pt = rotated + centroid_src
            reconstructed[name] = final_pt.tolist()
            
    expected_cb = [11.0, 11.0, 10.0]
    reconstructed_cb = reconstructed["CB"]
    diff = np.array(expected_cb) - np.array(reconstructed_cb)
    rmsd = np.sqrt(np.mean(diff**2))
    
    metrics = {
        "reconstructed": reconstructed,
        "rmsd": rmsd,
        "atoms_reconstructed": len(reconstructed)
    }
    
    with open("crates/proxide_fixer/tests/baseline.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Benchmark complete. RMSD: {rmsd}")

if __name__ == "__main__":
    benchmark()
