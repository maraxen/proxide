use pyo3::prelude::*;
use reqwest::blocking::Client;
use reqwest::StatusCode;
use std::fs::File;
use std::io::copy;
use std::path::Path;

const RCSB_URL_MMCIF: &str = "https://files.rcsb.org/download/";
const RCSB_URL_PDB: &str = "https://files.rcsb.org/download/";
const MDCATH_URL_BASE: &str = "http://mdcath.dat.s3-website-us-west-2.amazonaws.com/data/";
const AFDB_URL_BASE: &str = "https://alphafold.ebi.ac.uk/files/";

/// Fetch structure content from the RCSB data bank.
#[pyfunction]
#[pyo3(signature = (pdb_id, output_dir, format_type = "mmcif"))]
pub fn fetch_rcsb(pdb_id: &str, output_dir: &str, format_type: &str) -> PyResult<String> {
    let output_path = Path::new(output_dir);
    if !output_path.exists() {
        std::fs::create_dir_all(output_path)?;
    }

    let client = Client::new();
    let pdb_id_upper = pdb_id.to_uppercase();

    // Try primary format first
    let (url, filename) = match format_type {
        "pdb" => (
            format!("{}{}.pdb", RCSB_URL_PDB, pdb_id_upper),
            format!("{}.pdb", pdb_id_upper),
        ),
        _ => (
            format!("{}{}.cif", RCSB_URL_MMCIF, pdb_id_upper),
            format!("{}.cif", pdb_id_upper),
        ),
    };

    let target_path = output_path.join(&filename);
    if target_path.exists() {
        return Ok(target_path.to_string_lossy().to_string());
    }

    let _resp = match _fetch_with_retry(&client, &url) {
        Ok(r) => r,
        Err(_) => {
            // Fallback logic: if mmcif failed, try pdb, and vice versa
            let (fb_url, fb_filename) = if format_type == "pdb" {
                (
                    format!("{}{}.cif", RCSB_URL_MMCIF, pdb_id_upper),
                    format!("{}.cif", pdb_id_upper),
                )
            } else {
                (
                    format!("{}{}.pdb", RCSB_URL_PDB, pdb_id_upper),
                    format!("{}.pdb", pdb_id_upper),
                )
            };

            match _fetch_with_retry(&client, &fb_url) {
                Ok(r) => {
                    // Update target path if fallback succeeded
                    output_path.join(&fb_filename).to_string_lossy().to_string(); // Just to get path
                                                                                  // Consider logging warning here? usage of print! is generally discouraged in lib
                    r
                }
                Err(e) => {
                    return Err(pyo3::exceptions::PyIOError::new_err(format!(
                        "Failed to fetch {}: {}",
                        pdb_id, e
                    )))
                }
            }
        }
    };

    // We might have switched filenames in fallback, need to be careful.
    // Let's restructure slightly to be cleaner.
    // Actually, simpler logic:

    // 1. Try requested format
    let result = _download_file(&client, &url, &target_path);

    if result.is_ok() {
        return Ok(target_path.to_string_lossy().to_string());
    }

    // 2. Fallback
    let (fb_url, fb_filename) = if format_type == "pdb" {
        (
            format!("{}{}.cif", RCSB_URL_MMCIF, pdb_id_upper),
            format!("{}.cif", pdb_id_upper),
        )
    } else {
        (
            format!("{}{}.pdb", RCSB_URL_PDB, pdb_id_upper),
            format!("{}.pdb", pdb_id_upper),
        )
    };

    let fb_target_path = output_path.join(&fb_filename);
    match _download_file(&client, &fb_url, &fb_target_path) {
        Ok(_) => Ok(fb_target_path.to_string_lossy().to_string()),
        Err(e) => Err(pyo3::exceptions::PyIOError::new_err(format!(
            "Failed to fetch {} (both formats failed): {}",
            pdb_id, e
        ))),
    }
}

/// Fetch an h5 file from the MD-CATH data bank.
#[pyfunction]
pub fn fetch_md_cath(md_cath_id: &str, output_dir: &str) -> PyResult<String> {
    let output_path = Path::new(output_dir);
    if !output_path.exists() {
        std::fs::create_dir_all(output_path)?;
    }

    let filename = format!("{}.h5", md_cath_id);
    let target_path = output_path.join(&filename);

    if target_path.exists() {
        return Ok(target_path.to_string_lossy().to_string());
    }

    let subdirs = &md_cath_id[1..3];
    let url = format!("{}{}/{}.h5", MDCATH_URL_BASE, subdirs, md_cath_id);

    let client = Client::new();
    match _download_file(&client, &url, &target_path) {
        Ok(_) => Ok(target_path.to_string_lossy().to_string()),
        Err(e) => Err(pyo3::exceptions::PyIOError::new_err(format!(
            "Failed to fetch MD-CATH {}: {}",
            md_cath_id, e
        ))),
    }
}

/// Fetch a structure from the AlphaFold Structure Database (AFDB).
#[pyfunction]
#[pyo3(signature = (uniprot_id, output_dir, version = 4))]
pub fn fetch_afdb(uniprot_id: &str, output_dir: &str, version: u32) -> PyResult<String> {
    let output_path = Path::new(output_dir);
    if !output_path.exists() {
        std::fs::create_dir_all(output_path)?;
    }

    // Standard AFDB format: AF-{UNIPROT}-F1-model_v{VERSION}.pdb (or .cif)
    // We default to PDB format for AFDB as it is most common for simple usage, but they offer CIF too.
    // The previous implementation used .pdb, we stick to that.
    let filename = format!("AF-{}-F1-model_v{}.pdb", uniprot_id, version);
    let target_path = output_path.join(&filename);

    if target_path.exists() {
        return Ok(target_path.to_string_lossy().to_string());
    }

    let url = format!("{}{}", AFDB_URL_BASE, filename);
    let client = Client::new();

    match _download_file(&client, &url, &target_path) {
        Ok(_) => Ok(target_path.to_string_lossy().to_string()),
        Err(e) => Err(pyo3::exceptions::PyIOError::new_err(format!(
            "Failed to fetch AFDB {}: {}",
            uniprot_id, e
        ))),
    }
}

fn _fetch_with_retry(client: &Client, url: &str) -> Result<reqwest::blocking::Response, String> {
    let max_retries = 3;
    let mut delay = std::time::Duration::from_secs(1);

    for i in 0..max_retries {
        match client.get(url).send() {
            Ok(resp) => {
                if resp.status().is_success() {
                    return Ok(resp);
                } else if resp.status() == StatusCode::NOT_FOUND {
                    return Err(format!("404 Not Found"));
                }
                // Determine if we should retry based on status code
                // Generally 5xx errors are retryable
                if !resp.status().is_server_error() {
                    return Err(format!("Request failed with status: {}", resp.status()));
                }
            }
            Err(_) => {} // Network error, retry
        }

        if i < max_retries - 1 {
            std::thread::sleep(delay);
            delay *= 2;
        }
    }

    Err("Max retries exceeded".to_string())
}

fn _download_file(client: &Client, url: &str, target_path: &Path) -> Result<(), String> {
    let mut file = File::create(target_path).map_err(|e| e.to_string())?;
    _download_file_to_writer(client, url, &mut file)
}

fn _download_file_to_writer<W: std::io::Write>(
    client: &Client,
    url: &str,
    writer: &mut W,
) -> Result<(), String> {
    let mut response = _fetch_with_retry(client, url)?;
    copy(&mut response, writer).map_err(|e| e.to_string())?;
    Ok(())
}

/// Fetch a FoldComp database from the MMseqs2 server.
/// downloads: {db}, {db}.index, {db}.lookup, {db}.dbtype, {db}.source
#[pyfunction]
#[pyo3(signature = (db_name, output_dir, _download_chunks=16))]
pub fn fetch_foldcomp_database(
    db_name: &str,
    output_dir: &str,
    _download_chunks: usize,
) -> PyResult<String> {
    let output_path = Path::new(output_dir);
    if !output_path.exists() {
        std::fs::create_dir_all(output_path)?;
    }

    // Base URL: https://opendata.mmseqs.org/foldcomp/
    let base_url = "https://opendata.mmseqs.org/foldcomp/";
    let client = Client::new();

    let extensions = ["", ".index", ".lookup", ".dbtype", ".source"];

    for ext in extensions.iter() {
        let filename = format!("{}{}", db_name, ext);
        let url = format!("{}{}", base_url, filename);
        let target_path = output_path.join(&filename);

        if target_path.exists() {
            // Check size? For now skip if exists.
            continue;
        }

        // Use a temporary file to avoid partial downloads
        let tmp_path = target_path.with_extension("tmp");
        {
            let mut file = File::create(&tmp_path).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to create tmp file: {}", e))
            })?;

            _download_file_to_writer(&client, &url, &mut file).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!(
                    "Failed to download {}: {}",
                    filename, e
                ))
            })?;
        }

        std::fs::rename(&tmp_path, &target_path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to rename tmp file: {}", e))
        })?;
    }

    Ok(output_path.to_string_lossy().to_string())
}
