"""Command-line interface for Proxide."""

import pathlib
from typing import Optional

import typer
from rich.console import Console

from proxide.io import fetching

app = typer.Typer(
    name="proxide",
    help="Protein I/O for JAX with high-performance Rust backend",
    add_completion=False,
)

fetch_app = typer.Typer(help="Fetch structure data from online databases")
app.add_typer(fetch_app, name="fetch")

console = Console()


@fetch_app.command("rcsb")
def fetch_rcsb(
    pdb_id: str = typer.Argument(..., help="PDB ID (e.g., 1crn)"),
    output_dir: str = typer.Option(".", "--output-dir", "-o", help="Output directory"),
    format_type: str = typer.Option("mmcif", "--format", "-f", help="Format (pdb or mmcif)"),
):
    """Fetch structure content from the RCSB data bank."""
    try:
        path = fetching.fetch_rcsb(pdb_id, output_dir=output_dir, format_type=format_type)
        console.print(f"[green]Successfully fetched {pdb_id} to {path}[/green]")
    except Exception as e:
        console.print(f"[red]Error fetching {pdb_id}: {e}[/red]")
        raise typer.Exit(code=1)


@fetch_app.command("afdb")
def fetch_afdb(
    uniprot_id: str = typer.Argument(..., help="UniProt ID (e.g., P12345)"),
    output_dir: str = typer.Option(".", "--output-dir", "-o", help="Output directory"),
    version: int = typer.Option(4, "--version", "-v", help="AlphaFold version"),
):
    """Fetch a structure from the AlphaFold Structure Database (AFDB)."""
    try:
        path = fetching.fetch_afdb(uniprot_id, output_dir=output_dir, version=version)
        console.print(f"[green]Successfully fetched {uniprot_id} to {path}[/green]")
    except Exception as e:
        console.print(f"[red]Error fetching {uniprot_id}: {e}[/red]")
        raise typer.Exit(code=1)


@fetch_app.command("mdcath")
def fetch_md_cath(
    md_cath_id: str = typer.Argument(..., help="MD-CATH ID"),
    output_dir: str = typer.Option(".", "--output-dir", "-o", help="Output directory"),
):
    """Fetch an h5 file from the MD-CATH data bank."""
    try:
        path = fetching.fetch_md_cath(md_cath_id, output_dir=output_dir)
        console.print(f"[green]Successfully fetched {md_cath_id} to {path}[/green]")
    except Exception as e:
        console.print(f"[red]Error fetching {md_cath_id}: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("validate")
def validate(
    input_file: pathlib.Path = typer.Argument(..., help="Input structure file (PDB, MMCIF, FCZ)"),
):
    """Validate a structure file by attempting to parse it."""
    from proxide.io.parsing.backend import parse_structure
    
    try:
        if not input_file.exists():
            console.print(f"[red]Error: File {input_file} does not exist[/red]")
            raise typer.Exit(code=1)
            
        console.print(f"Validating {input_file}...")
        system = parse_structure(str(input_file))
        
        console.print(f"[green]Validation successful![/green]")
        console.print(f"  - Residues: {system.num_residues}")
        console.print(f"  - Atoms: {system.num_protein_atoms}")
        
        if hasattr(system, "chain_ids") and system.chain_ids is not None:
             console.print(f"  - Chains: {set(system.chain_ids)}")
             
    except Exception as e:
        console.print(f"[red]Validation failed: {e}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
