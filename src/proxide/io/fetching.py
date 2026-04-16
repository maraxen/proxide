"""Fetching utilities backed by the Rust extension."""

from proxide import _proxider


def fetch_rcsb(pdb_id: str, output_dir: str = ".", format_type: str = "mmcif") -> str:
  """Fetch structure content from the RCSB data bank.

  Args:
      pdb_id: The PDB ID to fetch.
      output_dir: Directory to save the file.
      format_type: Format to fetch ("mmcif" or "pdb").

  Returns:
      Path to the downloaded file.
  """
  return _proxider.fetch_rcsb(pdb_id, output_dir, format_type)


def fetch_md_cath(md_cath_id: str, output_dir: str = ".") -> str:
  """Fetch h5 content from the MD-CATH data bank.

  Args:
      md_cath_id: The MD-CATH ID.
      output_dir: Directory to save the file.

  Returns:
      Path to the downloaded file.
  """
  return _proxider.fetch_md_cath(md_cath_id, output_dir)


def fetch_afdb(uniprot_id: str, output_dir: str = ".", version: int = 4) -> str:
  """Fetch structure from AlphaFold Structure Database.

  Args:
      uniprot_id: The UniProt ID to fetch.
      output_dir: Directory to save the file.
      version: AlphaFold database version (default: 4).

  Returns:
      Path to the downloaded file.
  """
  return _proxider.fetch_afdb(uniprot_id, output_dir, version)


def fetch_foldcomp_database(db_name: str, output_dir: str = ".", download_chunks: int = 1) -> str:
  """Fetch a FoldComp database.

  Args:
      db_name: Name of the database (e.g., "afdb_swissprot").
      output_dir: Directory to save the database.
      download_chunks: Number of chunks to download in parallel.

  Returns:
      Path to the downloaded database directory.
  """
  return _proxider.fetch_foldcomp_database(db_name, output_dir, download_chunks)
