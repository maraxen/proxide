use std::path::Path;
use std::collections::HashMap;
use std::process::Command;
use thiserror::Error;

/// Error type for PROPKA3 invocation and parsing.
#[derive(Error, Debug)]
pub enum PropkaError {
    #[error("propka3 not found on PATH")]
    NotInstalled,

    #[error("failed to spawn propka3: {0}")]
    Spawn(String),

    #[error("propka3 exited with non-zero status: {0}")]
    NonZeroExit(String),

    #[error("failed to parse propka output: {0}")]
    Parse(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// pKa table: (chain_id, res_id) -> pKa value
pub type PkaTable = HashMap<(String, i32), f32>;

/// Run propka3 on a PDB file at the given pH.
///
/// This is a pure I/O boundary: it spawns propka3 as a subprocess,
/// writes a temp PDB file, collects the output, parses the .pka file,
/// and returns a table of pKa values indexed by (chain, res_id).
///
/// Does NOT mutate any topology. Returns Err on any failure (spawn, non-zero exit, parse).
/// Note: `ph` is not forwarded to propka3 (propka emits pKa independent of target pH);
/// the caller thresholds pKa against pH in assign_by_pka.
/// Only returns Ok(PkaTable) if propka3 succeeds and output is parseable.
pub fn run_propka(pdb_path: &Path, _ph: f32) -> Result<PkaTable, PropkaError> {
    // Try to spawn propka3
    let output = Command::new("propka3")
        .arg(pdb_path.to_string_lossy().as_ref())
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                PropkaError::NotInstalled
            } else {
                PropkaError::Spawn(e.to_string())
            }
        })?;

    // Check exit status
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(PropkaError::NonZeroExit(stderr.to_string()));
    }

    // propka3 writes <stem>.pka in the current working directory (or sometimes next to the input)
    // Try to locate it
    let pka_path = if let Some(stem) = pdb_path.file_stem() {
        pdb_path.parent().unwrap_or_else(|| Path::new(".")).join(format!("{}.pka", stem.to_string_lossy()))
    } else {
        return Err(PropkaError::Parse("Cannot determine pKa output filename".to_string()));
    };

    // Also try cwd just in case
    let pka_path = if pka_path.exists() {
        pka_path
    } else {
        let stem = pdb_path.file_stem().ok_or_else(|| {
            PropkaError::Parse("Cannot extract PDB filename stem".to_string())
        })?;
        std::env::current_dir()
            .ok()
            .map(|cwd| cwd.join(format!("{}.pka", stem.to_string_lossy())))
            .unwrap_or(pka_path)
    };

    // Read and parse the .pka file
    let pka_content = std::fs::read_to_string(&pka_path)
        .map_err(|e| PropkaError::Io(e))?;

    parse_pka_summary(&pka_content)
}

/// Parse the SUMMARY section of a .pka file.
///
/// Expected format (example):
/// ```text
/// SUMMARY OF THIS PREDICTION
///  Group                      pKa  model-pKa   ligand atom-type
/// ...
///  ASP    12 A                3.80       3.80      0.00    OD1
/// ...
/// ```
///
/// Returns a HashMap of (chain, res_id) -> pKa.
fn parse_pka_summary(content: &str) -> Result<PkaTable, PropkaError> {
    let mut table = PkaTable::new();

    // Find the SUMMARY section
    let summary_start = content
        .find("SUMMARY OF THIS PREDICTION")
        .ok_or_else(|| PropkaError::Parse("No SUMMARY section found".to_string()))?;

    let summary_end = content
        .get(summary_start..)
        .and_then(|s| s.find("\n\n"))
        .map(|pos| summary_start + pos)
        .unwrap_or(content.len());

    let summary_section = &content[summary_start..summary_end];

    for line in summary_section.lines() {
        // Skip header lines
        if line.contains("SUMMARY") || line.contains("Group") || line.contains("pKa") {
            continue;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Parse format: "RES_NAME    RES_ID CHAIN_ID    pKa ..."
        // Example: "ASP    12 A                3.80       3.80      0.00    OD1"
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() < 4 {
            continue; // Skip malformed lines
        }

        // parts[0] = residue name (ASP, GLU, HIS, etc.)
        // parts[1] = residue number
        // parts[2] = chain ID
        // parts[3] = pKa value

        let res_id_str = parts.get(1).copied().unwrap_or("");
        let chain_id = parts.get(2).copied().unwrap_or("");
        let pka_str = parts.get(3).copied().unwrap_or("");

        if let (Ok(res_id), Ok(pka)) = (res_id_str.parse::<i32>(), pka_str.parse::<f32>()) {
            table.insert((chain_id.to_string(), res_id), pka);
        }
    }

    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_pka_summary_basic() {
        let pka_content = r#"
SUMMARY OF THIS PREDICTION
 Group                      pKa  model-pKa   ligand atom-type
--------------------------------------------------------------------
 ASP    12 A                3.80       3.80      0.00    OD1
 GLU    45 A                4.50       4.50      0.00    OE1
 HIS    67 B                5.20       5.20      0.00    ND1

Prediction done.
"#;

        let table = parse_pka_summary(pka_content).expect("parse failed");
        assert_eq!(table.len(), 3);
        assert_eq!(table.get(&("A".to_string(), 12)), Some(&3.80));
        assert_eq!(table.get(&("A".to_string(), 45)), Some(&4.50));
        assert_eq!(table.get(&("B".to_string(), 67)), Some(&5.20));
    }

    #[test]
    fn test_parse_pka_summary_empty() {
        let pka_content = "SUMMARY OF THIS PREDICTION\n";
        let table = parse_pka_summary(pka_content).expect("parse failed");
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn test_run_propka_not_installed() {
        // If propka3 is not on PATH, this should return NotInstalled
        // (test may be skipped in CI if propka3 is installed)
        let result = run_propka(Path::new("/nonexistent/file.pdb"), 7.4);
        assert!(
            matches!(result, Err(PropkaError::NotInstalled)),
            "ENOENT must map to NotInstalled (propka3 absent in CI), got {:?}",
            result
        );
    }
}
