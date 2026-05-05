"""Performance benchmark comparing Rust vs JAX backends for Expaloma charge assignment."""

import time

from rdkit import Chem

from proxide.chem.partial_charges import assign_espaloma_charges_rdkit

# Test SMILES of varying complexity
BENCH_SMILES = [
    ("N2", "N#N"),
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ("Large", "C1=CC=C(C=C1)CC(C(=O)O)N"), # Phenylalanine
]

def run_benchmark():
    backends = ["jax", "rust"]
    iterations = 50

    print(f"{'Molecule':<12} | {'Backend':<8} | {'Avg Time (ms)':<15} | {'Speedup'}")
    print("-" * 55)

    for name, smiles in BENCH_SMILES:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        results = {}

        for backend in backends:
            # Warmup
            assign_espaloma_charges_rdkit(mol, backend=backend)

            start = time.perf_counter()
            for _ in range(iterations):
                assign_espaloma_charges_rdkit(mol, backend=backend)
            end = time.perf_counter()

            avg_ms = (end - start) * 1000 / iterations
            results[backend] = avg_ms

        speedup = results["jax"] / results["rust"]

        print(f"{name:<12} | jax      | {results['jax']:>13.3f} |")
        print(f"{'':<12} | rust     | {results['rust']:>13.3f} | {speedup:>7.1f}x")
        print("-" * 55)

if __name__ == "__main__":
    run_benchmark()
