//! Accession-set resolution: turning a requested list of accessions (today,
//! one per line from a text file; eventually FASTA-MSA headers and/or GTDB
//! shard membership) into the `wanted` filter passed to
//! [`crate::SketchStore::load_parquet`].

use crate::error::Result;
use std::path::Path;

/// Reads one accession per line, trimming whitespace and skipping blank
/// lines. This is the CLI's `--accessions <file>` input format.
pub fn read_accession_list(path: &Path) -> Result<Vec<String>> {
    let contents = std::fs::read_to_string(path)?;
    Ok(contents
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_and_trims_skipping_blanks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("accessions.txt");
        std::fs::write(&path, "GCA_000002425.3\n\n  GCA_000002426.1  \n").unwrap();

        let accessions = read_accession_list(&path).unwrap();
        assert_eq!(accessions, vec!["GCA_000002425.3", "GCA_000002426.1"]);
    }
}
