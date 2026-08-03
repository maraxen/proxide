#![cfg(feature = "fetching")]
// TODO: Review allow attributes at a later point
#![allow(clippy::useless_conversion)]
// ...

use reqwest::blocking::Client;
use reqwest::StatusCode;
use std::fs::File;
use std::io::copy;
use std::path::Path;

#[allow(dead_code)]
const RCSB_URL_MMCIF: &str = "https://files.rcsb.org/download/";
const RCSB_URL_PDB: &str = "https://files.rcsb.org/download/";
const MDCATH_URL_BASE: &str = "http://mdcath.dat.s3-website-us-west-2.amazonaws.com/data/";
const AFDB_URL_BASE: &str = "https://alphafold.ebi.ac.uk/files/";

/// Validates that an ID contains only safe characters for URL and path construction.
fn validate_id(id: &str) -> Result<(), String> {
    // TODO: For WASM targets, consider using Origin Private File System (OPFS)
    // to store and retrieve these downloaded files for high-performance synchronous access.
    if id.is_empty() {
        return Err("ID cannot be empty".to_string());
    }
    if id
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
    {
        Ok(())
    } else {
        Err(format!(
            "Invalid ID: '{}'. Only alphanumeric characters, underscores, and hyphens are allowed.",
            id
        ))
    }
}

/// Fetch structure content from the RCSB data bank.
pub fn fetch_rcsb(pdb_id: &str, output_dir: &str, format_type: &str) -> Result<String, String> {
    _fetch_rcsb_with_base(pdb_id, output_dir, format_type, RCSB_URL_PDB)
}

fn _fetch_rcsb_with_base(
    pdb_id: &str,
    output_dir: &str,
    format_type: &str,
    base_url: &str,
) -> Result<String, String> {
    validate_id(pdb_id)?;
    let output_path = Path::new(output_dir);
    if !output_path.exists() {
        std::fs::create_dir_all(output_path).map_err(|e| e.to_string())?;
    }

    let client = Client::new();
    let pdb_id_upper = pdb_id.to_uppercase();

    // Try primary format first
    let (url, filename) = match format_type {
        "pdb" => (
            format!("{}{}.pdb", base_url, pdb_id_upper),
            format!("{}.pdb", pdb_id_upper),
        ),
        _ => (
            format!("{}{}.cif", base_url, pdb_id_upper),
            format!("{}.cif", pdb_id_upper),
        ),
    };

    let target_path = output_path.join(&filename);
    if target_path.exists() {
        return Ok(target_path.to_string_lossy().to_string());
    }

    // 1. Try requested format
    let result = _download_file(&client, &url, &target_path);

    if result.is_ok() {
        return Ok(target_path.to_string_lossy().to_string());
    }

    // 2. Fallback
    let (fb_url, fb_filename) = if format_type == "pdb" {
        (
            format!("{}{}.cif", base_url, pdb_id_upper),
            format!("{}.cif", pdb_id_upper),
        )
    } else {
        (
            format!("{}{}.pdb", base_url, pdb_id_upper),
            format!("{}.pdb", pdb_id_upper),
        )
    };

    let fb_target_path = output_path.join(&fb_filename);
    match _download_file(&client, &fb_url, &fb_target_path) {
        Ok(_) => Ok(fb_target_path.to_string_lossy().to_string()),
        Err(e) => Err(format!(
            "Failed to fetch {} (both formats failed): {}",
            pdb_id, e
        )),
    }
}

/// Fetch an h5 file from the MD-CATH data bank.
pub fn fetch_md_cath(md_cath_id: &str, output_dir: &str) -> Result<String, String> {
    _fetch_md_cath_with_base(md_cath_id, output_dir, MDCATH_URL_BASE)
}

fn _fetch_md_cath_with_base(
    md_cath_id: &str,
    output_dir: &str,
    base_url: &str,
) -> Result<String, String> {
    validate_id(md_cath_id)?;
    let output_path = Path::new(output_dir);
    if !output_path.exists() {
        std::fs::create_dir_all(output_path).map_err(|e| e.to_string())?;
    }

    let filename = format!("{}.h5", md_cath_id);
    let target_path = output_path.join(&filename);

    if target_path.exists() {
        return Ok(target_path.to_string_lossy().to_string());
    }

    if md_cath_id.len() < 3 {
        return Err(format!(
            "Invalid MD-CATH ID: '{}'. ID must be at least 3 characters long for subdir sharding.",
            md_cath_id
        ));
    }
    let subdirs = &md_cath_id[1..3];
    let url = format!("{}{}/{}.h5", base_url, subdirs, md_cath_id);

    let client = Client::new();
    match _download_file(&client, &url, &target_path) {
        Ok(_) => Ok(target_path.to_string_lossy().to_string()),
        Err(e) => Err(format!("Failed to fetch MD-CATH {}: {}", md_cath_id, e)),
    }
}

/// Fetch a structure from the AlphaFold Structure Database (AFDB).
pub fn fetch_afdb(uniprot_id: &str, output_dir: &str, version: u32) -> Result<String, String> {
    _fetch_afdb_with_base(uniprot_id, output_dir, version, AFDB_URL_BASE)
}

fn _fetch_afdb_with_base(
    uniprot_id: &str,
    output_dir: &str,
    version: u32,
    base_url: &str,
) -> Result<String, String> {
    validate_id(uniprot_id)?;
    let output_path = Path::new(output_dir);
    if !output_path.exists() {
        std::fs::create_dir_all(output_path).map_err(|e| e.to_string())?;
    }

    let filename = format!("AF-{}-F1-model_v{}.pdb", uniprot_id, version);
    let target_path = output_path.join(&filename);

    if target_path.exists() {
        return Ok(target_path.to_string_lossy().to_string());
    }

    let url = format!("{}{}", base_url, filename);
    let client = Client::new();

    match _download_file(&client, &url, &target_path) {
        Ok(_) => Ok(target_path.to_string_lossy().to_string()),
        Err(e) => Err(format!("Failed to fetch AFDB {}: {}", uniprot_id, e)),
    }
}

fn _fetch_with_retry(client: &Client, url: &str) -> Result<reqwest::blocking::Response, String> {
    let max_retries = 3;
    let mut delay = std::time::Duration::from_secs(1);

    for i in 0..max_retries {
        if let Ok(resp) = client.get(url).send() {
            if resp.status().is_success() {
                return Ok(resp);
            } else if resp.status() == StatusCode::NOT_FOUND {
                return Err("404 Not Found".to_string());
            }
            if !resp.status().is_server_error() {
                return Err(format!("Request failed with status: {}", resp.status()));
            }
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
pub fn fetch_foldcomp_database(
    db_name: &str,
    output_dir: &str,
    _download_chunks: usize,
) -> Result<String, String> {
    _fetch_foldcomp_database_with_base(
        db_name,
        output_dir,
        _download_chunks,
        "https://opendata.mmseqs.org/foldcomp/",
    )
}

fn _fetch_foldcomp_database_with_base(
    db_name: &str,
    output_dir: &str,
    _download_chunks: usize,
    base_url: &str,
) -> Result<String, String> {
    validate_id(db_name)?;
    let output_path = Path::new(output_dir);
    if !output_path.exists() {
        std::fs::create_dir_all(output_path).map_err(|e| e.to_string())?;
    }

    let client = Client::new();

    let extensions = ["", ".index", ".lookup", ".dbtype", ".source"];

    for ext in extensions.iter() {
        let filename = format!("{}{}", db_name, ext);
        let url = format!("{}{}", base_url, filename);
        let target_path = output_path.join(&filename);

        if target_path.exists() {
            continue;
        }

        let tmp_path = target_path.with_extension("tmp");
        {
            let mut file =
                File::create(&tmp_path).map_err(|e| format!("Failed to create tmp file: {}", e))?;

            _download_file_to_writer(&client, &url, &mut file)
                .map_err(|e| format!("Failed to download {}: {}", filename, e))?;
        }

        std::fs::rename(&tmp_path, &target_path)
            .map_err(|e| format!("Failed to rename tmp file: {}", e))?;
    }

    Ok(output_path.to_string_lossy().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;

    #[test]
    fn test_validate_id() {
        assert!(validate_id("1abc").is_ok());
        assert!(validate_id("ID_with-hyphen").is_ok());
        assert!(validate_id("").is_err());
        assert!(validate_id("id with spaces").is_err());
        assert!(validate_id("id!@#").is_err());
        assert!(validate_id("ab/c").is_err());
        assert!(validate_id("a.b").is_err());
        assert!(validate_id("\\test").is_err());
    }

    #[test]
    fn test_fetch_md_cath_short_id() {
        assert!(fetch_md_cath("ab", "/tmp").is_err());
    }

    #[test]
    fn test_fetch_with_retry_404() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/test.pdb");
            then.status(404);
        });

        let client = Client::new();
        let url = server.url("/test.pdb");
        let res = _fetch_with_retry(&client, &url);
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), "404 Not Found");
        mock.assert();
    }

    #[test]
    fn test_fetch_rcsb_pdb_primary_success() {
        let server = MockServer::start();
        let pdb_id = "1ABC";
        let mock = server.mock(|when, then| {
            when.method(GET).path(format!("/{}.pdb", pdb_id));
            then.status(200).body("PDB CONTENT");
        });

        let output_dir = tempfile::tempdir().unwrap();
        let base_url = format!("{}/", server.base_url());
        let path = _fetch_rcsb_with_base(
            pdb_id,
            output_dir.path().to_str().unwrap(),
            "pdb",
            &base_url,
        )
        .unwrap();

        assert!(Path::new(&path).exists());
        assert!(path.ends_with("1ABC.pdb"));
        mock.assert();
    }

    #[test]
    fn test_fetch_rcsb_pdb_fallback_to_cif() {
        let server = MockServer::start();
        let pdb_id = "1ABC";

        // Mock 404 for PDB, 200 for CIF
        let mock_pdb = server.mock(|when, then| {
            when.method(GET).path(format!("/{}.pdb", pdb_id));
            then.status(404);
        });
        let mock_cif = server.mock(|when, then| {
            when.method(GET).path(format!("/{}.cif", pdb_id));
            then.status(200).body("CIF CONTENT");
        });

        let output_dir = tempfile::tempdir().unwrap();
        let base_url = format!("{}/", server.base_url());

        let path = _fetch_rcsb_with_base(
            pdb_id,
            output_dir.path().to_str().unwrap(),
            "pdb",
            &base_url,
        )
        .unwrap();

        assert!(path.ends_with("1ABC.cif"));
        assert!(Path::new(&path).exists());
        mock_pdb.assert();
        mock_cif.assert();
    }

    #[test]
    fn test_fetch_afdb() {
        let server = MockServer::start();
        let uniprot_id = "P12345";
        let mock = server.mock(|when, then| {
            when.method(GET)
                .path(format!("/AF-{}-F1-model_v1.pdb", uniprot_id));
            then.status(200).body("AFDB CONTENT");
        });

        let output_dir = tempfile::tempdir().unwrap();
        let base_url = format!("{}/", server.base_url());
        let path = _fetch_afdb_with_base(
            uniprot_id,
            output_dir.path().to_str().unwrap(),
            1,
            &base_url,
        )
        .unwrap();
        assert!(Path::new(&path).exists());
        mock.assert();
    }

    #[test]
    fn test_fetch_foldcomp_database() {
        let server = MockServer::start();
        let db_name = "testdb";
        let extensions = vec!["", ".index", ".lookup", ".dbtype", ".source"];
        let mut mocks = Vec::new();
        for ext in extensions {
            let path = format!("/{}{}", db_name, ext);
            mocks.push(server.mock(|when, then| {
                when.method(GET).path(path);
                then.status(200).body("CONTENT");
            }));
        }

        let output_dir = tempfile::tempdir().unwrap();
        let base_url = format!("{}/", server.base_url());
        let _path = _fetch_foldcomp_database_with_base(
            db_name,
            output_dir.path().to_str().unwrap(),
            0,
            &base_url,
        )
        .unwrap();

        for ext in vec!["", ".index", ".lookup", ".dbtype", ".source"] {
            assert!(output_dir
                .path()
                .join(format!("{}{}", db_name, ext))
                .exists());
        }
        for mock in mocks {
            mock.assert();
        }
    }
}
