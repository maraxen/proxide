//! Offsets-config tests that mutate process-global cwd and environment.
//!
//! They run as their own integration-test process so lib tests that open
//! `XtcReader` in parallel threads never observe `set_current_dir` /
//! `set_var` / `remove_var`. A mutex serializes the tests inside this binary.
//! Assertions go through the public `resolve_offsets_dir` and `XtcReader::open`
//! API; both were already public, so nothing was re-exported.

use proxide_io::formats::xtc::{resolve_offsets_dir, XtcReader};
use std::path::{Path, PathBuf};

static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn project_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn test_xtc_path() -> Option<PathBuf> {
    let path = project_root().join("tests/data/trajectories/test.xtc");
    path.exists().then_some(path)
}

/// Copy the checked-in fixture into a fresh tempdir so offset-cache sidecar
/// writes never touch the real fixture.
fn copy_fixture_to_tempdir(src: &Path) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("failed to create tempdir");
    let dst = dir.path().join(src.file_name().unwrap());
    std::fs::copy(src, &dst).expect("failed to copy fixture");
    (dir, dst)
}

struct EnvGuard {
    cwd: PathBuf,
    env: Option<String>,
    xdg: Option<String>,
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.cwd);
        match &self.env {
            Some(v) => std::env::set_var("PROXIDE_OFFSETS_DIR", v),
            None => std::env::remove_var("PROXIDE_OFFSETS_DIR"),
        }
        match &self.xdg {
            Some(v) => std::env::set_var("XDG_CONFIG_HOME", v),
            None => std::env::remove_var("XDG_CONFIG_HOME"),
        }
    }
}

#[test]
fn test_sidecar_resolver_5_tier_hierarchy() {
    let _lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _guard = EnvGuard {
        cwd: std::env::current_dir().unwrap(),
        env: std::env::var("PROXIDE_OFFSETS_DIR").ok(),
        xdg: std::env::var("XDG_CONFIG_HOME").ok(),
    };

    // Tier 1: Explicit path
    assert_eq!(
        resolve_offsets_dir(Some(Path::new("/tmp/explicit_cache"))).unwrap(),
        Some(PathBuf::from("/tmp/explicit_cache"))
    );
    // Explicit disable
    assert_eq!(resolve_offsets_dir(Some(Path::new("off"))).unwrap(), None);
    assert_eq!(resolve_offsets_dir(Some(Path::new("none"))).unwrap(), None);
    assert_eq!(resolve_offsets_dir(Some(Path::new(""))).unwrap(), None);

    // Tier 2: Environment variable
    std::env::set_var("PROXIDE_OFFSETS_DIR", "/tmp/env_cache");
    assert_eq!(
        resolve_offsets_dir(None).unwrap(),
        Some(PathBuf::from("/tmp/env_cache"))
    );
    // Tier 1 overrides Tier 2
    assert_eq!(
        resolve_offsets_dir(Some(Path::new("/tmp/explicit_override"))).unwrap(),
        Some(PathBuf::from("/tmp/explicit_override"))
    );
    // Env var disable
    std::env::set_var("PROXIDE_OFFSETS_DIR", "off");
    assert_eq!(resolve_offsets_dir(None).unwrap(), None);
    std::env::set_var("PROXIDE_OFFSETS_DIR", "none");
    assert_eq!(resolve_offsets_dir(None).unwrap(), None);
    std::env::remove_var("PROXIDE_OFFSETS_DIR");
}

/// Trailing comments, invalid TOML, wrong types, and unreadable configs must
/// not be stored as a path or treated as "unset". The walk stops at the
/// nearest `pyproject.toml`.
#[test]
fn test_offsets_dir_trailing_comment_and_invalid_config() {
    let _lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _guard = EnvGuard {
        cwd: std::env::current_dir().unwrap(),
        env: std::env::var("PROXIDE_OFFSETS_DIR").ok(),
        xdg: std::env::var("XDG_CONFIG_HOME").ok(),
    };
    std::env::remove_var("PROXIDE_OFFSETS_DIR");
    let xdg = tempfile::tempdir().unwrap();
    std::env::set_var("XDG_CONFIG_HOME", xdg.path());

    let commented = tempfile::tempdir().unwrap();
    std::fs::write(
        commented.path().join("pyproject.toml"),
        "[tool.proxide]\noffsets_dir = \"/fast/offsets\" # node-local\n",
    )
    .unwrap();
    std::env::set_current_dir(commented.path()).unwrap();
    assert_eq!(
        resolve_offsets_dir(None).expect("commented offsets_dir"),
        Some(PathBuf::from("/fast/offsets")),
        "a trailing TOML comment must not be part of the path"
    );

    let Some(xtc_src) = test_xtc_path() else {
        return;
    };
    let (_fix_dir, xtc_path) = copy_fixture_to_tempdir(&xtc_src);

    let invalid = tempfile::tempdir().unwrap();
    std::fs::write(
        invalid.path().join("pyproject.toml"),
        "[tool.proxide\noffsets_dir = \"/should-not-be-used\"\n",
    )
    .unwrap();
    std::env::set_current_dir(invalid.path()).unwrap();
    match XtcReader::open(&xtc_path) {
        Err(err) => assert!(
            err.to_string().contains("offsets config"),
            "open error must report the config failure, got {err}"
        ),
        Ok(_) => panic!("invalid pyproject.toml must fail the open"),
    }

    let wrong_type = tempfile::tempdir().unwrap();
    std::fs::write(
        wrong_type.path().join("pyproject.toml"),
        "[tool.proxide]\noffsets_dir = 12\n",
    )
    .unwrap();
    std::env::set_current_dir(wrong_type.path()).unwrap();
    let err = resolve_offsets_dir(None).expect_err("integer offsets_dir must fail");
    assert!(
        err.to_string().contains("string"),
        "wrong type must say the key has to be a string, got {err}"
    );
    match XtcReader::open(&xtc_path) {
        Err(err) => assert!(err.to_string().contains("offsets config"), "{err}"),
        Ok(_) => panic!("wrong-typed offsets_dir must fail the open"),
    }

    let ancestor = tempfile::tempdir().unwrap();
    std::fs::write(
        ancestor.path().join("pyproject.toml"),
        "[tool.proxide]\noffsets_dir = \"/from-parent\"\n",
    )
    .unwrap();
    let child = ancestor.path().join("child");
    std::fs::create_dir(&child).unwrap();
    std::fs::write(
        child.join("pyproject.toml"),
        "[project]\nname = \"child\"\n",
    )
    .unwrap();
    std::env::set_current_dir(&child).unwrap();
    assert_eq!(
        resolve_offsets_dir(None).expect("child pyproject is valid"),
        None,
        "nearest pyproject.toml without offsets_dir must not inherit an ancestor"
    );

    let alias = tempfile::tempdir().unwrap();
    std::fs::write(
        alias.path().join("pyproject.toml"),
        "[tool.proxide]\ncache_dir = \"/not-a-documented-key\"\n",
    )
    .unwrap();
    std::env::set_current_dir(alias.path()).unwrap();
    assert_eq!(
        resolve_offsets_dir(None).expect("undocumented key is not an error"),
        None,
        "only the documented offsets_dir key is honored"
    );

    let unreadable = tempfile::tempdir().unwrap();
    let cfg = unreadable.path().join("pyproject.toml");
    std::fs::write(&cfg, "[tool.proxide]\noffsets_dir = \"/fast/offsets\"\n").unwrap();
    let mut perms = std::fs::metadata(&cfg).unwrap().permissions();
    perms.set_readonly(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        perms.set_mode(0o000);
    }
    std::fs::set_permissions(&cfg, perms).unwrap();
    std::env::set_current_dir(unreadable.path()).unwrap();
    match XtcReader::open(&xtc_path) {
        Err(err) => assert!(err.to_string().contains("offsets config"), "{err}"),
        Ok(_) => panic!("an unreadable pyproject.toml must fail the open"),
    }

    // XDG tier: trailing comment, then an invalid file. No pyproject on the walk.
    let xdg_root = tempfile::tempdir().unwrap();
    let proxide_cfg = xdg_root.path().join("proxide");
    std::fs::create_dir(&proxide_cfg).unwrap();
    std::fs::write(
        proxide_cfg.join("config.toml"),
        "[offsets]\ndir = \"/xdg/offsets\" # node-local\n",
    )
    .unwrap();
    std::env::set_var("XDG_CONFIG_HOME", xdg_root.path());
    let bare = tempfile::tempdir().unwrap();
    std::env::set_current_dir(bare.path()).unwrap();
    assert_eq!(
        resolve_offsets_dir(None).expect("xdg comment"),
        Some(PathBuf::from("/xdg/offsets")),
        "XDG trailing comment must not be part of the path"
    );
    std::fs::write(proxide_cfg.join("config.toml"), "[offsets]\ndir = \n").unwrap();
    match XtcReader::open(&xtc_path) {
        Err(err) => assert!(err.to_string().contains("offsets config"), "{err}"),
        Ok(_) => panic!("invalid XDG config must fail the open"),
    }
}
