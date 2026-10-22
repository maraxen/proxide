"""Fetching utilities backed by the Rust extension."""

from proxide import _proxider


def fetch_rcsb(pdb_id: str, output_dir: str = ".", format_type: str = "mmcif") -> str:
  """Fetch structure content from the RCSB data bank.

  A cached file is reused only if it is complete. A truncated or empty copy is
  downloaded again. If the requested format returns 404, the other format is tried.
  Any other HTTP error is raised unchanged.

  Environment variables:
      PROXIDE_HTTP_CONNECT_TIMEOUT_SECS: connect timeout in seconds (default 30).
      PROXIDE_HTTP_TIMEOUT_SECS: per-read timeout in seconds, covering the wait for
          headers and each body read (default 120). Invalid values raise RuntimeError.
      PROXIDE_RCSB_BASE_URL: download host, for tests (a trailing slash is added).

  Args:
      pdb_id: The PDB ID to fetch.
      output_dir: Directory to save the file.
      format_type: Format to fetch ("mmcif" or "pdb").

  Returns:
      Path to the downloaded file.

  Raises:
      RuntimeError: on invalid ID, HTTP error, timeout, or a bad timeout setting.
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
