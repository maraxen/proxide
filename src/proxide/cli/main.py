"""Proxide command-line interface."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from proxide import (
  OutputSpec,
  __version__,
  fetch_afdb,
  fetch_foldcomp_database,
  fetch_rcsb,
  parse_structure,
)

app = typer.Typer(
  name="proxide",
  help="Proxide: High-performance protein I/O and processing for JAX.",
  add_completion=False,
)
fetch_app = typer.Typer(help="Fetch structure data from online databases")
app.add_typer(fetch_app, name="fetch")

console = Console()


@app.callback()
def callback():
  """Proxide: High-performance protein I/O for JAX."""
  pass


@fetch_app.command("rcsb")
def fetch_rcsb_cmd(
  id: str = typer.Argument(..., help="PDB ID (e.g., 1crn)"),
  output: str = typer.Option(".", "--output", "-o", help="Output directory"),
  format: str = typer.Option("mmcif", "--format", "-f", help="Format (pdb or mmcif)"),
):
  """Fetch structure from the RCSB Protein Data Bank."""
  try:
    with console.status(f"[bold green]Fetching {id} from RCSB..."):
      path = fetch_rcsb(id, output, format_type=format)
    console.print(f"[bold green]✓[/bold green] Downloaded to: [cyan]{path}[/cyan]")
  except Exception as e:
    console.print(f"[red]Error fetching {id}: {e}[/red]")
    raise typer.Exit(1) from e


@fetch_app.command("afdb")
def fetch_afdb_cmd(
  id: str = typer.Argument(..., help="UniProt ID (e.g., P12345)"),
  output: str = typer.Option(".", "--output", "-o", help="Output directory"),
  version: int = typer.Option(4, "--version", "-v", help="AlphaFold version"),
):
  """Fetch structure from the AlphaFold Structure Database (AFDB)."""
  try:
    with console.status(f"[bold green]Fetching {id} from AFDB..."):
      path = fetch_afdb(id, output, version=version)
    console.print(f"[bold green]✓[/bold green] Downloaded to: [cyan]{path}[/cyan]")
  except Exception as e:
    console.print(f"[red]Error fetching {id}: {e}[/red]")
    raise typer.Exit(1) from e


@fetch_app.command("foldcomp")
def fetch_foldcomp_cmd(
  db: str = typer.Argument(..., help="FoldComp database name"),
  output: str = typer.Option(".", "--output", "-o", help="Output directory"),
):
  """Fetch a FoldComp database."""
  try:
    with console.status(f"[bold green]Fetching FoldComp DB {db}..."):
      path = fetch_foldcomp_database(db, output)
    console.print(f"[bold green]✓[/bold green] Downloaded to: [cyan]{path}[/cyan]")
  except Exception as e:
    console.print(f"[red]Error fetching {db}: {e}[/red]")
    raise typer.Exit(1) from e


@app.command()
def info(
  path: str = typer.Argument(..., help="Path to structure file (PDB, mmCIF, etc.)"),
):
  """Display summary information about a structure file."""
  try:
    with console.status("[bold blue]Analyzing structure..."):
      protein = parse_structure(path)

    table = Table(title=f"Structure Summary: {Path(path).name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    # Use internal attributes from Protein container
    n_res = protein.coordinates.shape[0] if hasattr(protein, "coordinates") else 0

    table.add_row("Number of Residues", str(n_res))
    table.add_row("Number of Chains", str(len(np.unique(protein.chain_index)) if hasattr(protein, "chain_index") else "N/A"))
    table.add_row("Format", str(getattr(protein, "format", "Unknown")))

    if getattr(protein, "source", None):
      table.add_row("Source", protein.source)

    console.print(table)

  except Exception as e:
    console.print(f"[red]Error reading {path}: {e}[/red]")
    raise typer.Exit(1) from e


@app.command()
def convert(
  input: str = typer.Argument(..., help="Input file path"),
  output: str = typer.Argument(..., help="Output file path"),
  add_hydrogens: bool = typer.Option(False, "--add-hydrogens", "-a", help="Add missing hydrogens"),
  infer_bonds: bool = typer.Option(True, "--infer-bonds", "-i", help="Infer bond connectivity"),
  force_field: str | None = typer.Option(None, "--force-field", "-f", help="Force field for parameterization"),
):
  """Convert structure between formats and apply modifications."""
  from proxide.io.parsing.backend import iterload, write_dcd
  from proxide.io.writing import write_mmcif, write_npz, write_pdb

  try:
    in_path = Path(input)
    out_path = Path(output)

    # Detect trajectory formats for streaming
    traj_exts = {".xtc", ".dcd", ".trr"}
    is_traj_in = in_path.suffix.lower() in traj_exts
    is_traj_out = out_path.suffix.lower() in traj_exts

    if is_traj_in and is_traj_out:
      with console.status(f"[bold green]Streaming trajectory [cyan]{input}[/cyan] -> [cyan]{output}[/cyan]..."):
        # Initial analysis to get number of atoms
        # For now, we use the first chunk to detect metadata
        stream = iterload(input, chunk_size=100)

        writer = None
        total_frames = 0

        for chunk in stream:
          coords = chunk["coordinates"]  # (chunk_size, N, 3)
          n_frames_chunk, n_atoms, _ = coords.shape

          if writer is None:
            if out_path.suffix.lower() == ".dcd":
              writer = write_dcd(output, n_atoms)
            else:
              # Fallback or error if writer not implemented
              console.print(f"[red]Error: Writing to {out_path.suffix} is not yet supported in streaming mode.[/red]")
              raise typer.Exit(1)

          # Write frames in chunk
          for i in range(n_frames_chunk):
            writer.write_frame(coords[i].flatten().tolist(), None)

          total_frames += n_frames_chunk

        if writer:
          writer.finalize()

      console.print(f"[bold green]✓[/bold green] Streamed {total_frames} frames from [cyan]{input}[/cyan] to [cyan]{output}[/cyan]")
      return

    # Standard structure conversion
    spec = OutputSpec(
      add_hydrogens=add_hydrogens,
      infer_bonds=infer_bonds,
      parameterize_md=bool(force_field),
      force_field=force_field,
    )

    with console.status("[bold green]Processing structure..."):
      protein = parse_structure(input, spec)

    if out_path.suffix.lower() == ".pdb":
      write_pdb(protein, output)
    elif out_path.suffix.lower() == ".npz":
      write_npz(protein, output)
    elif out_path.suffix.lower() in [".cif", ".mmcif"]:
      write_mmcif(protein, output)
    else:
      console.print(f"[yellow]Warning: Format {out_path.suffix} not yet supported natively. Saving as NPZ.[/yellow]")
      write_npz(protein, out_path.with_suffix(".npz"))

    console.print(f"[bold green]✓[/bold green] Converted [cyan]{input}[/cyan] to [cyan]{output}[/cyan]")

  except Exception as e:
    console.print(f"[red]Error during conversion: {e}[/red]")
    raise typer.Exit(1) from e


@app.command()
def validate(
  path: str = typer.Argument(..., help="File to validate"),
):
  """Check if a file can be parsed correctly by Proxide."""
  try:
    with console.status(f"[bold blue]Validating {path}..."):
      parse_structure(path)
    console.print(f"[bold green]✓[/bold green] {path} is valid.")
  except Exception as e:
    console.print(f"[bold red]✗[/bold red] Validation failed for {path}: {e}")
    raise typer.Exit(1) from e


@app.command()
def bench(
  path: str = typer.Argument(..., help="Benchmark target file"),
  iterations: int = typer.Option(10, "--iterations", "-n", help="Number of iterations"),
):
  """Benchmark parsing performance on the specified file."""
  try:
    console.print(f"[bold]Benchmarking [cyan]{path}[/cyan] over {iterations} iterations...[/bold]")

    times = []
    for _ in range(iterations):
      start = time.perf_counter()
      parse_structure(path)
      times.append(time.perf_counter() - start)

    avg = np.mean(times)
    std = np.std(times)

    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Average Time", f"{avg*1000:.2f} ± {std*1000:.2f} ms")
    table.add_row("Best Time", f"{np.min(times)*1000:.2f} ms")
    table.add_row("Throughput", f"{1.0/avg:.2f} files/s")

    console.print(table)

  except Exception as e:
    console.print(f"[red]Benchmark failed: {e}[/red]")
    raise typer.Exit(1) from e


@app.command()
def parameterize(
  input: str = typer.Argument(..., help="Input molecule file (MOL2, SDF)"),
  output: str = typer.Option(None, "--output", "-o", help="Output .npz path (defaults to stem.npz)"),
):
  """Assign GAFF parameters to a small molecule and save to NPZ."""
  from proxide.io.parsing.molecule import Molecule

  try:
    if output is None:
      output = str(Path(input).with_suffix(".npz"))

    with console.status(f"[bold green]Parameterizing {input}..."):
      if input.lower().endswith(".mol2"):
        mol = Molecule.from_mol2(input)
      elif input.lower().endswith(".sdf"):
        mol = Molecule.from_sdf(input)
      else:
        console.print("[red]Error: Supported molecule formats are .mol2 and .sdf[/red]")
        raise typer.Exit(1)

      mol.parameterize()

      # Prepare data for NPZ
      data = {
        "positions": mol.positions,
        "atom_types": mol.atom_types,
        "charges": mol.charges,
        "elements": mol.elements,
        "bonds": np.array(mol.bonds),
      }

      np.savez_compressed(output, **data)

    console.print(f"[bold green]✓[/bold green] Parameterized and saved to [cyan]{output}[/cyan]")

  except Exception as e:
    console.print(f"[red]Parameterization failed: {e}[/red]")
    raise typer.Exit(1) from e


@app.command()
def charges(
  input: str = typer.Argument(..., help="Input molecule file (MOL2, SDF)"),
  backend: str = typer.Option("rust", "--backend", "-b", help="Backend for computation: 'rust' or 'jax'"),
):
  """Assign partial charges via Expaloma AM1-BCC surrogate."""
  from proxide.chem.partial_charges import assign_espaloma_charges_from_proxide_molecule
  from proxide.io.parsing.molecule import Molecule

  try:
    with console.status(f"[bold green]Assigning charges using backend: {backend}..."):
      if input.lower().endswith(".mol2"):
        mol = Molecule.from_mol2(input)
      elif input.lower().endswith(".sdf"):
        mol = Molecule.from_sdf(input)
      else:
        console.print("[red]Error: Supported molecule formats are .mol2 and .sdf[/red]")
        raise typer.Exit(1)

      start = time.perf_counter()
      q = assign_espaloma_charges_from_proxide_molecule(mol, backend=backend)
      elapsed = time.perf_counter() - start

    console.print(f"[bold green]✓[/bold green] Assigned {len(q)} partial charges in [cyan]{elapsed*1000:.2f} ms[/cyan]")

  except Exception as e:
    console.print(f"[red]Charge assignment failed: {e}[/red]")
    raise typer.Exit(1) from e


@app.command()
def version():
  """Print the version of Proxide."""
  console.print(f"Proxide [bold cyan]{__version__}[/bold cyan]")


if __name__ == "__main__":
  app()
