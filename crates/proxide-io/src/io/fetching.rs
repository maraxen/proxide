#![cfg(feature = "fetching")]
// TODO: Review allow attributes at a later point
#![allow(clippy::useless_conversion)]
// ...

use reqwest::blocking::Client;
use reqwest::StatusCode;
use std::fs::{File, OpenOptions};
use std::io::{copy, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Once;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const RCSB_URL_PDB: &str = "https://files.rcsb.org/download/";
const MDCATH_URL_BASE: &str = "http://mdcath.dat.s3-website-us-west-2.amazonaws.com/data/";
const AFDB_URL_BASE: &str = "https://alphafold.ebi.ac.uk/files/";
const FOLDCOMP_URL_BASE: &str = "https://opendata.mmseqs.org/foldcomp/";

/// Overrides the RCSB download host. A missing trailing slash is added.
/// Used by the test suite to point at a local server; leave unset in production.
const RCSB_BASE_URL_ENV: &str = "PROXIDE_RCSB_BASE_URL";
/// Connect timeout in seconds (default 30).
const CONNECT_TIMEOUT_ENV: &str = "PROXIDE_HTTP_CONNECT_TIMEOUT_SECS";
/// Per-read timeout in seconds (default 120). reqwest 0.11 blocking applies it to
/// the connect/headers wait and to each body read, not to the whole body.
const REQUEST_TIMEOUT_ENV: &str = "PROXIDE_HTTP_TIMEOUT_SECS";
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 30;
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 120;
/// Sent on every request. AFDB has returned 403 to clients without an identifying User-Agent.
const USER_AGENT: &str = concat!(
    "proxide/",
    env!("CARGO_PKG_VERSION"),
    " (+https://github.com/maraxen/proxide)"
);
/// Attempts for a request that gets a 5xx response or a connection error.
const MAX_ATTEMPTS: u32 = 3;
/// Bytes read from the end of a cached PDB or mmCIF file to check it is complete.
const TAIL_BYTES: u64 = 4096;
/// Partial downloads older than this are removed when a new download for the same file starts.
const STALE_PARTIAL_AGE: Duration = Duration::from_secs(60 * 60);

/// Gives each partial download in this process a distinct name.
static PARTIAL_SEQ: AtomicU64 = AtomicU64::new(0);
static BASE_URL_LOGGED: Once = Once::new();

/// Failure of one download. Only `NotFound` (HTTP 404) lets the RCSB format fallback run.
#[derive(Debug)]
enum FetchError {
    NotFound(String),
    Failed(String),
}

impl FetchError {
    fn is_not_found(&self) -> bool {
        matches!(self, FetchError::NotFound(_))
    }
}

impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FetchError::NotFound(msg) | FetchError::Failed(msg) => f.write_str(msg),
        }
    }
}

/// Parses a timeout in seconds. `None` means the default. Zero and non-numbers are errors.
fn parse_timeout_secs(
    name: &str,
    raw: Option<&str>,
    default_secs: u64,
) -> Result<Duration, String> {
    let Some(raw) = raw else {
        return Ok(Duration::from_secs(default_secs));
    };
    match raw.trim().parse::<u64>() {
        Ok(secs) if secs > 0 => Ok(Duration::from_secs(secs)),
        _ => Err(format!(
            "{} must be a positive whole number of seconds, got {:?}",
            name, raw
        )),
    }
}

fn timeout_from_env(name: &str, default_secs: u64) -> Result<Duration, String> {
    let raw = match std::env::var_os(name) {
        None => None,
        Some(value) => Some(
            value
                .into_string()
                .map_err(|_| format!("{} is not valid unicode", name))?,
        ),
    };
    parse_timeout_secs(name, raw.as_deref(), default_secs)
}

/// Per-read timeout for the public fetch functions.
fn request_timeout() -> Result<Duration, String> {
    timeout_from_env(REQUEST_TIMEOUT_ENV, DEFAULT_REQUEST_TIMEOUT_SECS)
}

fn http_client(timeout: Duration) -> Result<Client, String> {
    let connect = timeout_from_env(CONNECT_TIMEOUT_ENV, DEFAULT_CONNECT_TIMEOUT_SECS)?;
    Client::builder()
        .user_agent(USER_AGENT)
        .connect_timeout(connect)
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
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        Ok(())
    } else {
        Err(format!(
            "Invalid ID: '{}'. Only ASCII letters, digits, underscores, and hyphens are allowed.",
            id
        ))
    }
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

/// True for a regular file with at least one byte.
fn is_cached(path: &Path) -> bool {
    std::fs::metadata(path)
        .map(|m| m.is_file() && m.len() > 0)
        .unwrap_or(false)
}

/// Last `TAIL_BYTES` of a file, decoded lossily.
fn read_tail(path: &Path) -> std::io::Result<String> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    file.seek(SeekFrom::Start(len.saturating_sub(TAIL_BYTES)))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// A cached file can be reused only if it is complete.
///
/// `.pdb` files must contain an exact `END` record line in their last `TAIL_BYTES`.
/// `ENDMDL` does not count: a multi-model file cut right after a model ends with it.
/// `.cif` files must end with a `#` line (checked on RCSB 1CRN, 4HHB and 6VXX).
/// Other formats (h5, FoldComp) only need to be non-empty. A truncated copy from an
/// older build is not complete and is downloaded again.
fn cache_is_complete(path: &Path) -> bool {
    if !is_cached(path) {
        return false;
    }
    match path.extension().and_then(|e| e.to_str()) {
        Some("pdb") => match read_tail(path) {
            Ok(tail) => tail.lines().any(|line| line.trim_end() == "END"),
            Err(_) => false,
        },
        Some("cif") => match read_tail(path) {
            Ok(tail) => tail
                .lines()
                .rev()
                .find(|line| !line.trim().is_empty())
                .map(|line| line.trim() == "#")
                .unwrap_or(false),
            Err(_) => false,
        },
        _ => true,
    }
}

/// Fetch structure content from the RCSB data bank.
///
/// Returns the path of a complete file. An existing file is reused only if it is
/// complete (see `cache_is_complete`). The format fallback runs only when the
/// requested format returns HTTP 404.
///
/// Environment variables:
/// - `PROXIDE_RCSB_BASE_URL`: download host (a trailing `/` is added if missing).
/// - `PROXIDE_HTTP_CONNECT_TIMEOUT_SECS` (default 30) and `PROXIDE_HTTP_TIMEOUT_SECS`
///   (default 120, per read). An invalid value is an error.
pub fn fetch_rcsb(pdb_id: &str, output_dir: &str, format_type: &str) -> Result<String, String> {
    let base = rcsb_base_url();
    _fetch_rcsb_with_base(pdb_id, output_dir, format_type, &base, request_timeout()?)
}

fn rcsb_base_url() -> String {
    match std::env::var(RCSB_BASE_URL_ENV)
        .ok()
        .filter(|s| !s.is_empty())
    {
        Some(mut url) => {
            if !url.ends_with('/') {
                url.push('/');
            }
            BASE_URL_LOGGED.call_once(|| {
                log::info!("{} is set; RCSB downloads use {}", RCSB_BASE_URL_ENV, url);
            });
            url
        }
        None => RCSB_URL_PDB.to_string(),
    }
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

    let (primary_ext, fallback_ext) = if format_type == "pdb" {
        ("pdb", "cif")
    } else {
        ("cif", "pdb")
    };

    let primary_path = output_path.join(format!("{}.{}", pdb_id_upper, primary_ext));
    if cache_is_complete(&primary_path) {
        return Ok(path_string(&primary_path));
    }
    let primary_url = format!("{}{}.{}", base_url, pdb_id_upper, primary_ext);

    match _download_file(&client, &primary_url, &primary_path) {
        Ok(()) => Ok(path_string(&primary_path)),
        Err(primary_err) if !primary_err.is_not_found() => {
            Err(format!("Failed to fetch {}: {}", pdb_id, primary_err))
        }
        Err(primary_err) => {
            // HTTP 404: this format does not exist for the entry, so try the other one.
            let fallback_path = output_path.join(format!("{}.{}", pdb_id_upper, fallback_ext));
            if cache_is_complete(&fallback_path) {
                return Ok(path_string(&fallback_path));
            }
            let fallback_url = format!("{}{}.{}", base_url, pdb_id_upper, fallback_ext);
            _download_file(&client, &fallback_url, &fallback_path)
                .map(|()| path_string(&fallback_path))
                .map_err(|fallback_err| {
                    format!(
                        "Failed to fetch {} (both formats failed): {}; {}",
                        pdb_id, primary_err, fallback_err
                    )
                })
        }
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

    let client = http_client(request_timeout()?)?;
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
    if cache_is_complete(&target_path) {
        return Ok(path_string(&target_path));
    }

    let url = format!("{}{}", base_url, filename);
    let client = http_client(request_timeout()?)?;
    _download_file(&client, &url, &target_path)
        .map_err(|e| format!("Failed to fetch AFDB {}: {}", uniprot_id, e))?;
    Ok(path_string(&target_path))
}

/// GET with bounded retries for 5xx responses and connection errors. Timeouts and
/// other 4xx responses are not retried. Errors name the URL and the status or cause.
fn _fetch_with_retry(
    client: &Client,
    url: &str,
) -> Result<reqwest::blocking::Response, FetchError> {
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
                if status == StatusCode::NOT_FOUND {
                    return Err(FetchError::NotFound(last_error));
                }
                if !status.is_server_error() {
                    return Err(FetchError::Failed(last_error));
                }
            }
            Err(e) if e.is_timeout() => {
                return Err(FetchError::Failed(format!("GET {} timed out: {}", url, e)));
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

    Err(FetchError::Failed(last_error))
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

/// Creates a new temporary file next to the target (same directory, so rename is atomic).
/// The name carries the pid, a nanosecond clock and a sequence number. `create_new`
/// makes a collision an error rather than a silent overwrite of another writer's file.
fn create_partial(parent: &Path, file_name: &str) -> Result<(PathBuf, File), String> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = PARTIAL_SEQ.fetch_add(1, Ordering::Relaxed);
    create_partial_with_token(parent, file_name, &format!("{:x}{:x}", nanos, seq))
}

fn partial_path(parent: &Path, file_name: &str, token: &str) -> PathBuf {
    parent.join(format!(
        ".{}.{}.{}.part",
        file_name,
        std::process::id(),
        token
    ))
}

fn create_partial_with_token(
    parent: &Path,
    file_name: &str,
    token: &str,
) -> Result<(PathBuf, File), String> {
    let tmp_path = partial_path(parent, file_name, token);
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp_path)
        .map_err(|e| format!("failed to create {}: {}", tmp_path.display(), e))?;
    Ok((tmp_path, file))
}

/// Removes `.{file_name}.*.part` files in `dir` that have not been written for an hour.
/// These are left behind by killed processes; live downloads keep their mtime fresh.
fn sweep_stale_partials(dir: &Path, file_name: &str) {
    let prefix = format!(".{}.", file_name);
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let now = SystemTime::now();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !(name.starts_with(&prefix) && name.ends_with(".part")) {
            continue;
        }
        let stale = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| now.duration_since(t).ok())
            .is_some_and(|age| age > STALE_PARTIAL_AGE);
        if stale {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Download `url` into `target_path` atomically.
///
/// The body goes to a temporary file in the same directory and is renamed into place
/// only after a complete, non-empty response. Any failure removes the temporary file,
/// so `target_path` is either absent, previously complete, or a complete download.
/// If the rename fails because another process already finished a complete copy,
/// that copy is accepted.
fn _download_file(client: &Client, url: &str, target_path: &Path) -> Result<(), FetchError> {
    let parent = target_path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = target_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    sweep_stale_partials(parent, &file_name);

    let (tmp_path, mut file) = create_partial(parent, &file_name).map_err(FetchError::Failed)?;
    let mut guard = PartialFile {
        path: tmp_path.clone(),
        committed: false,
    };

    let written = _download_file_to_writer(client, url, &mut file)?;
    if written == 0 {
        return Err(FetchError::Failed(format!(
            "empty response body from {}",
            url
        )));
    }
    file.sync_all().map_err(|e| {
        FetchError::Failed(format!("failed to flush {}: {}", tmp_path.display(), e))
    })?;
    drop(file);

    match std::fs::rename(&tmp_path, target_path) {
        Ok(()) => {
            guard.committed = true;
            Ok(())
        }
        Err(e) => {
            if cache_is_complete(target_path) {
                return Ok(());
            }
            Err(FetchError::Failed(format!(
                "failed to move download into {}: {}",
                target_path.display(),
                e
            )))
        }
    }
}

fn _download_file_to_writer<W: Write>(
    client: &Client,
    url: &str,
    writer: &mut W,
) -> Result<u64, FetchError> {
    let mut response = _fetch_with_retry(client, url)?;
    copy(&mut response, writer)
        .map_err(|e| FetchError::Failed(format!("failed reading body from {}: {}", url, e)))
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

    let client = http_client(request_timeout()?)?;

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
    use std::time::Instant;

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

    /// A PDB body that is complete: it ends with an END record.
    const COMPLETE_PDB: &str = "HEADER    TEST\nATOM      1  N   ALA A   1\nEND\n";
    /// A CIF body that is complete: it ends with a "#" line.
    const COMPLETE_CIF: &str = "data_1ABC\n_atom_site.id\nATOM 1\n#\n";

    /// Serves one connection. It writes `head_and_partial`, stalls for `stall`, then closes.
    fn serve_partial_then_stall(head_and_partial: &'static [u8], stall: Duration) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut request = [0u8; 4096];
            let _ = stream.read(&mut request);
            let _ = stream.write_all(head_and_partial);
            std::thread::sleep(stall);
        });
        format!("http://{}/", addr)
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
        assert!(
            validate_id("1ab\u{e9}").is_err(),
            "non-ASCII IDs are rejected"
        );
    }

    #[test]
    fn test_parse_timeout_secs() {
        let name = "PROXIDE_HTTP_TIMEOUT_SECS";
        assert_eq!(
            parse_timeout_secs(name, None, 120).unwrap(),
            Duration::from_secs(120)
        );
        assert_eq!(
            parse_timeout_secs(name, Some("45"), 120).unwrap(),
            Duration::from_secs(45)
        );
        assert_eq!(
            parse_timeout_secs(name, Some(" 7 "), 120).unwrap(),
            Duration::from_secs(7)
        );
        for bad in ["0", "abc", "", "-5", "1.5"] {
            let err = parse_timeout_secs(name, Some(bad), 120).unwrap_err();
            assert!(err.contains(name), "error should name the variable: {err}");
        }
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
        assert!(err.is_not_found());
        let msg = err.to_string();
        assert!(msg.contains("404"), "error should carry the status: {msg}");
        assert!(msg.contains(&url), "error should name the URL: {msg}");
        mock.assert();
    }

    #[test]
    fn test_fetch_rcsb_pdb_primary_success_writes_full_body() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/1ABC.pdb");
            then.status(200).body(COMPLETE_PDB);
        });

        let output_dir = tempfile::tempdir().unwrap();
        let path = _fetch_rcsb_with_base(
            "1ABC",
            output_dir.path().to_str().unwrap(),
            "pdb",
            &base(&server),
            TEST_TIMEOUT,
        )
        .unwrap();

        assert!(path.ends_with("1ABC.pdb"));
        assert_eq!(std::fs::read(&path).unwrap(), COMPLETE_PDB.as_bytes());
        assert_eq!(dir_entries(output_dir.path()), vec!["1ABC.pdb"]);
        mock.assert();
    }

    #[test]
    fn test_fetch_rcsb_pdb_404_falls_back_to_cif() {
        let server = MockServer::start();
        let mock_pdb = server.mock(|when, then| {
            when.method(GET).path("/1ABC.pdb");
            then.status(404);
        });
        let mock_cif = server.mock(|when, then| {
            when.method(GET).path("/1ABC.cif");
            then.status(200).body(COMPLETE_CIF);
        });

        let output_dir = tempfile::tempdir().unwrap();
        let path = _fetch_rcsb_with_base(
            "1ABC",
            output_dir.path().to_str().unwrap(),
            "pdb",
            &base(&server),
            TEST_TIMEOUT,
        )
        .unwrap();

        assert!(path.ends_with("1ABC.cif"));
        assert_eq!(std::fs::read(&path).unwrap(), COMPLETE_CIF.as_bytes());
        assert_eq!(dir_entries(output_dir.path()), vec!["1ABC.cif"]);
        mock_pdb.assert();
        mock_cif.assert();
    }

    #[test]
    fn test_fetch_rcsb_primary_500_does_not_fall_back() {
        let server = MockServer::start();
        let mock_pdb = server.mock(|when, then| {
            when.method(GET).path("/1ABC.pdb");
            then.status(500);
        });
        let mock_cif = server.mock(|when, then| {
            when.method(GET).path("/1ABC.cif");
            then.status(200).body(COMPLETE_CIF);
        });

        let output_dir = tempfile::tempdir().unwrap();
        let err = _fetch_rcsb_with_base(
            "1ABC",
            output_dir.path().to_str().unwrap(),
            "pdb",
            &base(&server),
            TEST_TIMEOUT,
        )
        .unwrap_err();

        assert!(err.contains("500"), "error should name the 500: {err}");
        mock_cif.assert_hits(0);
        assert!(mock_pdb.hits() >= 1);
        assert_eq!(dir_entries(output_dir.path()), Vec::<String>::new());
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
            then.status(200)
                .body(COMPLETE_PDB)
                .delay(Duration::from_secs(4));
        });
        let _cif = server.mock(|when, then| {
            when.method(GET).path("/1STL.cif");
            then.status(200)
                .body(COMPLETE_CIF)
                .delay(Duration::from_secs(4));
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
    fn test_fetch_rcsb_stall_mid_body_is_error_and_leaves_no_file() {
        // Headers and a partial body arrive, then the server goes silent past the per-read timeout.
        let head: &'static [u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 100000\r\nConnection: close\r\n\r\nHEADER    PARTIAL\nATOM      1  N";
        let base_url = serve_partial_then_stall(head, Duration::from_secs(4));

        let output_dir = tempfile::tempdir().unwrap();
        let started = Instant::now();
        let result = _fetch_rcsb_with_base(
            "1PRT",
            output_dir.path().to_str().unwrap(),
            "pdb",
            &base_url,
            TEST_TIMEOUT,
        );

        assert!(result.is_err(), "a stalled body must surface as an error");
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "per-read timeout not enforced"
        );
        assert_eq!(
            dir_entries(output_dir.path()),
            Vec::<String>::new(),
            "no target and no .part"
        );
    }

    #[test]
    fn test_fetch_rcsb_empty_cached_file_is_redownloaded() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/1ABC.pdb");
            then.status(200).body(COMPLETE_PDB);
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

        assert_eq!(std::fs::read(&path).unwrap(), COMPLETE_PDB.as_bytes());
        mock.assert();
    }

    #[test]
    fn test_fetch_rcsb_truncated_pdb_cache_is_redownloaded() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/1ABC.pdb");
            then.status(200).body(COMPLETE_PDB);
        });

        // A 1000-byte truncated file from an older build: non-empty but no END record.
        let output_dir = tempfile::tempdir().unwrap();
        let truncated =
            "ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N\n"
                .repeat(13);
        assert!(truncated.len() >= 1000 && !truncated.contains("END"));
        std::fs::write(output_dir.path().join("1ABC.pdb"), truncated.as_bytes()).unwrap();

        let path = _fetch_rcsb_with_base(
            "1ABC",
            output_dir.path().to_str().unwrap(),
            "pdb",
            &base(&server),
            TEST_TIMEOUT,
        )
        .unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), COMPLETE_PDB.as_bytes());
        mock.assert();
    }

    #[test]
    fn test_fetch_rcsb_truncated_cif_cache_is_redownloaded() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/1ABC.cif");
            then.status(200).body(COMPLETE_CIF);
        });

        let output_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            output_dir.path().join("1ABC.cif"),
            b"data_1ABC\n_atom_site.id\nATOM 1\nATOM 2\n",
        )
        .unwrap();

        let path = _fetch_rcsb_with_base(
            "1ABC",
            output_dir.path().to_str().unwrap(),
            "cif",
            &base(&server),
            TEST_TIMEOUT,
        )
        .unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), COMPLETE_CIF.as_bytes());
        mock.assert();
    }

    #[test]
    fn test_fetch_rcsb_complete_cache_is_a_hit_without_request() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET);
            then.status(500);
        });

        let output_dir = tempfile::tempdir().unwrap();
        std::fs::write(output_dir.path().join("1ABC.pdb"), COMPLETE_PDB.as_bytes()).unwrap();
        std::fs::write(output_dir.path().join("2ABC.cif"), COMPLETE_CIF.as_bytes()).unwrap();

        let pdb = _fetch_rcsb_with_base(
            "1ABC",
            output_dir.path().to_str().unwrap(),
            "pdb",
            &base(&server),
            TEST_TIMEOUT,
        )
        .unwrap();
        let cif = _fetch_rcsb_with_base(
            "2ABC",
            output_dir.path().to_str().unwrap(),
            "cif",
            &base(&server),
            TEST_TIMEOUT,
        )
        .unwrap();

        assert_eq!(std::fs::read(&pdb).unwrap(), COMPLETE_PDB.as_bytes());
        assert_eq!(std::fs::read(&cif).unwrap(), COMPLETE_CIF.as_bytes());
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
    fn test_sweep_removes_only_stale_partials_for_the_target() {
        let dir = tempfile::tempdir().unwrap();
        let stale = dir.path().join(".1ABC.pdb.111.1a.part");
        let fresh = dir.path().join(".1ABC.pdb.222.2b.part");
        let other = dir.path().join(".1XYZ.pdb.333.3c.part");
        let target = dir.path().join("1ABC.pdb");
        for p in [&stale, &fresh, &other, &target] {
            std::fs::write(p, b"x").unwrap();
        }
        let old = SystemTime::now() - Duration::from_secs(2 * 60 * 60);
        std::fs::File::options()
            .write(true)
            .open(&stale)
            .unwrap()
            .set_modified(old)
            .unwrap();

        sweep_stale_partials(dir.path(), "1ABC.pdb");

        assert!(!stale.exists(), "stale partial should be removed");
        assert!(fresh.exists(), "fresh partial must be kept");
        assert!(other.exists(), "partials of other targets must be kept");
        assert!(target.exists());
    }

    #[test]
    fn test_pdb_truncated_after_endmdl_is_redownloaded() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/1ABC.pdb");
            then.status(200).body(COMPLETE_PDB);
        });

        let output_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            output_dir.path().join("1ABC.pdb"),
            b"MODEL        1\nATOM      1  N   ALA A   1\nENDMDL\n",
        )
        .unwrap();

        let path = _fetch_rcsb_with_base(
            "1ABC",
            output_dir.path().to_str().unwrap(),
            "pdb",
            &base(&server),
            TEST_TIMEOUT,
        )
        .unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), COMPLETE_PDB.as_bytes());
        mock.assert();
    }

    #[test]
    fn test_pdb_truncated_mid_model_after_an_endmdl_is_redownloaded() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/1ABC.pdb");
            then.status(200).body(COMPLETE_PDB);
        });

        let output_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            output_dir.path().join("1ABC.pdb"),
            b"MODEL        1\nATOM      1  N   ALA A   1\nENDMDL\nMODEL        2\nATOM      1  N   ALA A   1\n",
        )
        .unwrap();

        let path = _fetch_rcsb_with_base(
            "1ABC",
            output_dir.path().to_str().unwrap(),
            "pdb",
            &base(&server),
            TEST_TIMEOUT,
        )
        .unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), COMPLETE_PDB.as_bytes());
        mock.assert();
    }

    #[test]
    fn test_complete_multi_model_pdb_is_a_cache_hit() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET);
            then.status(500);
        });

        let multi = b"MODEL        1\nATOM      1  N   ALA A   1\nENDMDL\nMODEL        2\nATOM      1  N   ALA A   1\nENDMDL\nMASTER      0    0    0    0    0    0    0    0    2    0    0    0\nEND   \n";
        let output_dir = tempfile::tempdir().unwrap();
        std::fs::write(output_dir.path().join("1ABC.pdb"), multi).unwrap();

        let path = _fetch_rcsb_with_base(
            "1ABC",
            output_dir.path().to_str().unwrap(),
            "pdb",
            &base(&server),
            TEST_TIMEOUT,
        )
        .unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), multi);
        assert_eq!(mock.hits(), 0);
    }

    #[test]
    fn test_create_partial_collision_errors_and_keeps_other_file() {
        let dir = tempfile::tempdir().unwrap();
        let other = partial_path(dir.path(), "1ABC.pdb", "deadbeef");
        std::fs::write(&other, b"OTHER WRITER").unwrap();

        let err = create_partial_with_token(dir.path(), "1ABC.pdb", "deadbeef").unwrap_err();

        assert!(err.contains("failed to create"), "unexpected error: {err}");
        assert_eq!(std::fs::read(&other).unwrap(), b"OTHER WRITER");
    }

    #[test]
    fn test_requests_send_user_agent() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET)
                .path("/1ABC.pdb")
                .header("user-agent", USER_AGENT);
            then.status(200).body(COMPLETE_PDB);
        });

        let output_dir = tempfile::tempdir().unwrap();
        _fetch_rcsb_with_base(
            "1ABC",
            output_dir.path().to_str().unwrap(),
            "pdb",
            &base(&server),
            TEST_TIMEOUT,
        )
        .unwrap();

        mock.assert();
    }

    #[test]
    fn test_fetch_afdb() {
        let server = MockServer::start();
        let uniprot_id = "P12345";
        let mock = server.mock(|when, then| {
            when.method(GET)
                .path(format!("/AF-{}-F1-model_v1.pdb", uniprot_id));
            then.status(200).body(COMPLETE_PDB);
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

        for ext in extensions {
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
