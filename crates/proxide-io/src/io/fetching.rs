#![cfg(feature = "fetching")]
// TODO: Review allow attributes at a later point
#![allow(clippy::useless_conversion)]
// ...

use reqwest::blocking::Client;
use std::fs::File;
use std::io::copy;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[allow(dead_code)]
const RCSB_URL_MMCIF: &str = "https://files.rcsb.org/download/";
const RCSB_URL_PDB: &str = "https://files.rcsb.org/download/";
const MDCATH_URL_BASE: &str = "http://mdcath.dat.s3-website-us-west-2.amazonaws.com/data/";
const AFDB_URL_BASE: &str = "https://alphafold.ebi.ac.uk/files/";
const FOLDCOMP_URL_BASE: &str = "https://opendata.mmseqs.org/foldcomp/";

/// Environment variable that overrides the RCSB download host. The Python test
/// suite points it at a local HTTP server; leave it unset in production.
const RCSB_BASE_URL_ENV: &str = "PROXIDE_RCSB_BASE_URL";

/// Connect timeout applied to every HTTP request.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Overall budget for one request: connect, response headers and body.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
/// Attempts for a request that gets a 5xx response or a connection error.
const MAX_ATTEMPTS: u32 = 3;

/// Distinguishes partial-download names within one process.
static PARTIAL_SEQ: AtomicU64 = AtomicU64::new(0);

fn http_client(timeout: Duration) -> Result<Client, String> {
    Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(timeout)
        .build()
        .map_err(|e| format!("failed to build HTTP client: {e}"))
}

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

fn path_string(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

/// A cached file counts as a hit only when it is a regular, non-empty file.
/// Empty files are left behind by interrupted or failed downloads.
fn is_cached(path: &Path) -> bool {
    std::fs::metadata(path)
        .map(|m| m.is_file() && m.len() > 0)
        .unwrap_or(false)
}

/// Fetch structure content from the RCSB data bank.
///
/// Returns the path of a complete, non-empty file. `PROXIDE_RCSB_BASE_URL`
/// overrides the download host (used by tests).
pub fn fetch_rcsb(pdb_id: &str, output_dir: &str, format_type: &str) -> Result<String, String> {
    let base = std::env::var(RCSB_BASE_URL_ENV)
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| RCSB_URL_PDB.to_string());
    _fetch_rcsb_with_base(pdb_id, output_dir, format_type, &base, REQUEST_TIMEOUT)
}

fn _fetch_rcsb_with_base(
    pdb_id: &str,
    output_dir: &str,
    format_type: &str,
    base_url: &str,
    timeout: Duration,
) -> Result<String, String> {
    validate_id(pdb_id)?;
    let output_path = Path::new(output_dir);
    std::fs::create_dir_all(output_path).map_err(|e| e.to_string())?;

    let client = http_client(timeout)?;
    let pdb_id_upper = pdb_id.to_uppercase();

    // Primary format first, then the other one.
    let (primary_ext, fallback_ext) = if format_type == "pdb" {
        ("pdb", "cif")
    } else {
        ("cif", "pdb")
    };
    let primary_path = output_path.join(format!("{}.{}", pdb_id_upper, primary_ext));
    if is_cached(&primary_path) {
        return Ok(path_string(&primary_path));
    }

    let primary_url = format!("{}{}.{}", base_url, pdb_id_upper, primary_ext);
    let primary_err = match _download_file(&client, &primary_url, &primary_path) {
        Ok(()) => return Ok(path_string(&primary_path)),
        Err(e) => e,
    };

    let fallback_path = output_path.join(format!("{}.{}", pdb_id_upper, fallback_ext));
    if is_cached(&fallback_path) {
        return Ok(path_string(&fallback_path));
    }
    let fallback_url = format!("{}{}.{}", base_url, pdb_id_upper, fallback_ext);
    match _download_file(&client, &fallback_url, &fallback_path) {
        Ok(()) => Ok(path_string(&fallback_path)),
        Err(fallback_err) => Err(format!(
            "Failed to fetch {} (both formats failed): {}; {}",
            pdb_id, primary_err, fallback_err
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
    std::fs::create_dir_all(output_path).map_err(|e| e.to_string())?;

    let target_path = output_path.join(format!("{}.h5", md_cath_id));
    if is_cached(&target_path) {
        return Ok(path_string(&target_path));
    }

    if md_cath_id.len() < 3 {
        return Err(format!(
            "Invalid MD-CATH ID: '{}'. ID must be at least 3 characters long for subdir sharding.",
            md_cath_id
        ));
    }
    let subdirs = &md_cath_id[1..3];
    let url = format!("{}{}/{}.h5", base_url, subdirs, md_cath_id);

    let client = http_client(REQUEST_TIMEOUT)?;
    _download_file(&client, &url, &target_path)
        .map_err(|e| format!("Failed to fetch MD-CATH {}: {}", md_cath_id, e))?;
    Ok(path_string(&target_path))
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
    std::fs::create_dir_all(output_path).map_err(|e| e.to_string())?;

    let filename = format!("AF-{}-F1-model_v{}.pdb", uniprot_id, version);
    let target_path = output_path.join(&filename);
    if is_cached(&target_path) {
        return Ok(path_string(&target_path));
    }

    let url = format!("{}{}", base_url, filename);
    let client = http_client(REQUEST_TIMEOUT)?;
    _download_file(&client, &url, &target_path)
        .map_err(|e| format!("Failed to fetch AFDB {}: {}", uniprot_id, e))?;
    Ok(path_string(&target_path))
}

/// GET with a bounded retry. Timeouts are not retried (each one already costs the
/// full request budget). Errors name the URL and the HTTP status or transport cause.
fn _fetch_with_retry(client: &Client, url: &str) -> Result<reqwest::blocking::Response, String> {
    let mut delay = Duration::from_secs(1);
    let mut last_error = String::new();

    for attempt in 1..=MAX_ATTEMPTS {
        match client.get(url).send() {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    return Ok(resp);
                }
                last_error = format!("GET {} returned HTTP {}", url, status);
                if !status.is_server_error() {
                    return Err(last_error);
                }
            }
            Err(e) if e.is_timeout() => {
                return Err(format!("GET {} timed out: {}", url, e));
            }
            Err(e) => {
                last_error = format!("GET {} failed: {}", url, e);
            }
        }

        if attempt < MAX_ATTEMPTS {
            std::thread::sleep(delay);
            delay *= 2;
        }
    }

    Err(last_error)
}

/// Removes a partial download on drop unless the download was committed.
struct PartialFile {
    path: PathBuf,
    committed: bool,
}

impl Drop for PartialFile {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Download `url` into `target_path` atomically.
///
/// The body is written to a temporary file in the same directory and renamed into
/// place only after a complete, non-empty response. Any failure removes the
/// temporary file, so `target_path` is either absent or a complete download.
fn _download_file(client: &Client, url: &str, target_path: &Path) -> Result<(), String> {
    let parent = target_path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = target_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let seq = PARTIAL_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp_path = parent.join(format!(
        ".{}.{}.{}.part",
        file_name,
        std::process::id(),
        seq
    ));

    let mut guard = PartialFile {
        path: tmp_path.clone(),
        committed: false,
    };
    {
        let mut file = File::create(&tmp_path)
            .map_err(|e| format!("failed to create {}: {}", tmp_path.display(), e))?;
        let written = _download_file_to_writer(client, url, &mut file)?;
        if written == 0 {
            return Err(format!("empty response body from {}", url));
        }
        file.sync_all()
            .map_err(|e| format!("failed to flush {}: {}", tmp_path.display(), e))?;
    }

    std::fs::rename(&tmp_path, target_path).map_err(|e| {
        format!(
            "failed to move download into {}: {}",
            target_path.display(),
            e
        )
    })?;
    guard.committed = true;
    Ok(())
}

fn _download_file_to_writer<W: std::io::Write>(
    client: &Client,
    url: &str,
    writer: &mut W,
) -> Result<u64, String> {
    let mut response = _fetch_with_retry(client, url)?;
    copy(&mut response, writer).map_err(|e| format!("failed reading body from {}: {}", url, e))
}

/// Fetch a FoldComp database from the MMseqs2 server.
pub fn fetch_foldcomp_database(
    db_name: &str,
    output_dir: &str,
    _download_chunks: usize,
) -> Result<String, String> {
    _fetch_foldcomp_database_with_base(db_name, output_dir, _download_chunks, FOLDCOMP_URL_BASE)
}

fn _fetch_foldcomp_database_with_base(
    db_name: &str,
    output_dir: &str,
    _download_chunks: usize,
    base_url: &str,
) -> Result<String, String> {
    validate_id(db_name)?;
    let output_path = Path::new(output_dir);
    std::fs::create_dir_all(output_path).map_err(|e| e.to_string())?;

    let client = http_client(REQUEST_TIMEOUT)?;

    for ext in ["", ".index", ".lookup", ".dbtype", ".source"] {
        let filename = format!("{}{}", db_name, ext);
        let target_path = output_path.join(&filename);
        if is_cached(&target_path) {
            continue;
        }
        let url = format!("{}{}", base_url, filename);
        _download_file(&client, &url, &target_path)
            .map_err(|e| format!("Failed to download {}: {}", filename, e))?;
    }

    Ok(path_string(output_path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use std::time::{Duration, Instant};

    const TEST_TIMEOUT: Duration = Duration::from_secs(1);

    fn base(server: &MockServer) -> String {
        format!("{}/", server.base_url())
    }

    fn dir_entries(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

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
    fn test_fetch_with_retry_404_names_url_and_status() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/test.pdb");
            then.status(404);
        });

        let client = http_client(TEST_TIMEOUT).unwrap();
        let url = server.url("/test.pdb");
        let err = _fetch_with_retry(&client, &url).unwrap_err();
        assert!(err.contains("404"), "error should carry the status: {err}");
        assert!(err.contains(&url), "error should name the URL: {err}");
        mock.assert();
    }

    #[test]
    fn test_fetch_rcsb_pdb_primary_success_writes_full_body() {
        let server = MockServer::start();
        let pdb_id = "1ABC";
        let mock = server.mock(|when, then| {
            when.method(GET).path(format!("/{}.pdb", pdb_id));
            then.status(200).body("PDB CONTENT");
        });

        let output_dir = tempfile::tempdir().unwrap();
        let path = _fetch_rcsb_with_base(
            pdb_id,
            output_dir.path().to_str().unwrap(),
            "pdb",
            &base(&server),
            TEST_TIMEOUT,
        )
        .unwrap();

        assert!(path.ends_with("1ABC.pdb"));
        assert_eq!(std::fs::read(&path).unwrap(), b"PDB CONTENT");
        assert_eq!(dir_entries(output_dir.path()), vec!["1ABC.pdb"]);
        mock.assert();
    }

    #[test]
    fn test_fetch_rcsb_pdb_fallback_to_cif() {
        let server = MockServer::start();
        let pdb_id = "1ABC";

        let mock_pdb = server.mock(|when, then| {
            when.method(GET).path(format!("/{}.pdb", pdb_id));
            then.status(404);
        });
        let mock_cif = server.mock(|when, then| {
            when.method(GET).path(format!("/{}.cif", pdb_id));
            then.status(200).body("CIF CONTENT");
        });

        let output_dir = tempfile::tempdir().unwrap();
        let path = _fetch_rcsb_with_base(
            pdb_id,
            output_dir.path().to_str().unwrap(),
            "pdb",
            &base(&server),
            TEST_TIMEOUT,
        )
        .unwrap();

        assert!(path.ends_with("1ABC.cif"));
        assert_eq!(std::fs::read(&path).unwrap(), b"CIF CONTENT");
        assert_eq!(dir_entries(output_dir.path()), vec!["1ABC.cif"]);
        mock_pdb.assert();
        mock_cif.assert();
    }

    #[test]
    fn test_fetch_rcsb_404_in_both_formats_is_error_and_leaves_no_file() {
        let server = MockServer::start();
        let _pdb = server.mock(|when, then| {
            when.method(GET).path("/9ZZZ.pdb");
            then.status(404);
        });
        let _cif = server.mock(|when, then| {
            when.method(GET).path("/9ZZZ.cif");
            then.status(404);
        });

        let output_dir = tempfile::tempdir().unwrap();
        let err = _fetch_rcsb_with_base(
            "9ZZZ",
            output_dir.path().to_str().unwrap(),
            "pdb",
            &base(&server),
            TEST_TIMEOUT,
        )
        .unwrap_err();

        assert!(err.contains("404"), "error should carry the status: {err}");
        assert!(err.contains("9ZZZ"), "error should name the ID: {err}");
        assert_eq!(dir_entries(output_dir.path()), Vec::<String>::new());
    }

    #[test]
    fn test_fetch_rcsb_stall_past_timeout_is_error_and_leaves_no_file() {
        let server = MockServer::start();
        let _pdb = server.mock(|when, then| {
            when.method(GET).path("/1STL.pdb");
            then.status(200).body("LATE").delay(Duration::from_secs(4));
        });
        let _cif = server.mock(|when, then| {
            when.method(GET).path("/1STL.cif");
            then.status(200).body("LATE").delay(Duration::from_secs(4));
        });

        let output_dir = tempfile::tempdir().unwrap();
        let started = Instant::now();
        let result = _fetch_rcsb_with_base(
            "1STL",
            output_dir.path().to_str().unwrap(),
            "pdb",
            &base(&server),
            TEST_TIMEOUT,
        );

        assert!(result.is_err(), "a stalled server must surface as an error");
        assert!(
            started.elapsed() < Duration::from_secs(8),
            "timeout not enforced"
        );
        assert_eq!(dir_entries(output_dir.path()), Vec::<String>::new());
    }

    #[test]
    fn test_fetch_rcsb_empty_cached_file_is_redownloaded() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/1ABC.pdb");
            then.status(200).body("PDB CONTENT");
        });

        let output_dir = tempfile::tempdir().unwrap();
        std::fs::write(output_dir.path().join("1ABC.pdb"), b"").unwrap();

        let path = _fetch_rcsb_with_base(
            "1ABC",
            output_dir.path().to_str().unwrap(),
            "pdb",
            &base(&server),
            TEST_TIMEOUT,
        )
        .unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), b"PDB CONTENT");
        mock.assert();
    }

    #[test]
    fn test_fetch_rcsb_nonempty_cached_file_is_a_hit_without_request() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET);
            then.status(500);
        });

        let output_dir = tempfile::tempdir().unwrap();
        std::fs::write(output_dir.path().join("1ABC.pdb"), b"CACHED").unwrap();

        let path = _fetch_rcsb_with_base(
            "1ABC",
            output_dir.path().to_str().unwrap(),
            "pdb",
            &base(&server),
            TEST_TIMEOUT,
        )
        .unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), b"CACHED");
        assert_eq!(mock.hits(), 0);
    }

    #[test]
    fn test_fetch_rcsb_empty_response_body_is_error_and_leaves_no_file() {
        let server = MockServer::start();
        let _pdb = server.mock(|when, then| {
            when.method(GET).path("/1EMP.pdb");
            then.status(200).body("");
        });
        let _cif = server.mock(|when, then| {
            when.method(GET).path("/1EMP.cif");
            then.status(200).body("");
        });

        let output_dir = tempfile::tempdir().unwrap();
        let result = _fetch_rcsb_with_base(
            "1EMP",
            output_dir.path().to_str().unwrap(),
            "pdb",
            &base(&server),
            TEST_TIMEOUT,
        );

        assert!(result.is_err());
        assert_eq!(dir_entries(output_dir.path()), Vec::<String>::new());
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
        let extensions = ["", ".index", ".lookup", ".dbtype", ".source"];
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

        for ext in ["", ".index", ".lookup", ".dbtype", ".source"] {
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
