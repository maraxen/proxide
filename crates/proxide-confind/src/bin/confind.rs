//! `confind` command line driver.
//!
//! Required: `--p <pdb>` `--rLib <library>`.
//! Optional: `--o <file>`, `--sel <spec>`, `--cdcut <f64>`, `--incut <f64>`, `--threads <n>`,
//! and the configuration options `--cbclash`, `--crowdmode`, `--freedommode`, `--scope`,
//! `--dclash`, `--dcont`, `--dcut`, `--taulo`, `--tauhi` (see the crate README).
//! Flags may appear in any order. An invalid value or configuration exits non-zero with a
//! message naming the option, the offending value and the allowed values.

use orx_parallel::{IntoParIter, ParIter};
use proxide_confind::{
    load_pdb_f64, CbClashPolicy, ConFind, ConFindConfig, ConFindError, ConfigError,
    CrowdednessMode, Cutoffs, FreedomMode, ProteinBackbone, ResidueIndex, ResidueScope,
};
use proxide_rotlib::RotamerLibrary;
use std::fs;
use std::io::{BufWriter, Write};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let opts = parse_args(&args)?;

    log::info!("loading rotamer library: {}", opts.rlib.display());
    let rotlib = Arc::new(RotamerLibrary::load_pb(&opts.rlib)?);

    log::info!("loading PDB: {}", opts.pdb.display());
    let backbone = Arc::new(load_pdb_f64(&opts.pdb)?);

    // Resolve the freedom scope against the loaded structure.
    let scope = match &opts.scope {
        ScopeArg::All => ResidueScope::All,
        ScopeArg::Spec(spec) => ResidueScope::Subset(
            parse_selection(spec, &backbone).map_err(|e| format!("--scope: {e}"))?,
        ),
    };
    let config = opts.config.clone().with_scope(scope);
    if let Some(n) = config.threads {
        log::info!("configured parallel thread count: {n}");
    }

    let cf = ConFind::with_config(rotlib, backbone.clone(), false, config)
        .map_err(|e| describe_error(&e))?;

    // Determine residue selection.
    let residues: Vec<ResidueIndex> = if let Some(ref sel) = opts.sel {
        parse_selection(sel, &backbone).map_err(|e| format!("--sel: {e}"))?
    } else {
        (0..backbone.bb.len() as u32).map(ResidueIndex).collect()
    };

    log::info!("running contacts on {} residues...", residues.len());
    let contact_list = cf.contacts(&residues, opts.cd_cut)?;

    let crowdedness_list: Vec<(ResidueIndex, f64)> = residues
        .iter()
        .map(|&ri| cf.crowdedness(ri).map(|c| (ri, c)))
        .collect::<Result<_, _>>()?;

    // Freedom is reported for selected residues inside the analysis scope.
    let freedom_residues: Vec<ResidueIndex> =
        residues.iter().copied().filter(|&ri| cf.in_scope(ri)).collect();
    let par = freedom_residues.into_par();
    let par = match cf.config().threads {
        Some(n) => par.num_threads(n.get()),
        None => par,
    };
    let freedom_list: Vec<(ResidueIndex, f64)> = par
        .map(|ri| cf.freedom(ri).map(|f| (ri, f)))
        .collect::<Vec<Result<(ResidueIndex, f64), ConFindError>>>()
        .into_iter()
        .collect::<Result<_, _>>()?;

    let interference_list = cf.interference(&residues, opts.in_cut)?;

    // Write output.
    let out: Box<dyn Write> = if let Some(ref path) = opts.out {
        Box::new(BufWriter::new(fs::File::create(path)?))
    } else {
        Box::new(BufWriter::new(std::io::stdout()))
    };
    let mut out = out;

    // Header: every effective option value, so a run is reproducible from its output.
    let cfg = cf.config();
    let cut = &cfg.cutoffs;
    writeln!(out, "# cbclash\t{}", cfg.cb_clash_policy)?;
    writeln!(out, "# crowdmode\t{}", cfg.crowdedness_mode)?;
    writeln!(out, "# freedommode\t{}", cfg.freedom_mode)?;
    match &opts.scope {
        ScopeArg::All => writeln!(out, "# scope\tall")?,
        ScopeArg::Spec(spec) => {
            let n = match &cfg.scope {
                ResidueScope::Subset(v) => v.len(),
                ResidueScope::All => backbone.bb.len(),
            };
            writeln!(out, "# scope\t{spec}\t({n} residues)")?
        }
    }
    writeln!(out, "# dclash\t{}", cut.d_clash)?;
    writeln!(out, "# dcont\t{}", cut.d_cont)?;
    writeln!(out, "# dcut\t{}", cut.d_cut)?;
    writeln!(out, "# taulo\t{}", cut.tau_lo)?;
    writeln!(out, "# tauhi\t{}", cut.tau_hi)?;
    writeln!(out, "# cdcut\t{}", opts.cd_cut)?;
    writeln!(out, "# incut\t{}", opts.in_cut)?;

    for (&(ri, rj), &cd) in contact_list.pairs.iter().zip(&contact_list.degrees) {
        let id_a = cf.residue_id(ri);
        let id_b = cf.residue_id(rj);
        writeln!(
            out,
            "contact\t{},{}\t{},{}\t{:.6}\t{}\t{}",
            id_a.chain_id,
            format_resnum(id_a.res_id, id_a.insertion_code),
            id_b.chain_id,
            format_resnum(id_b.res_id, id_b.insertion_code),
            cd,
            backbone.bb[ri.0 as usize].res_name,
            backbone.bb[rj.0 as usize].res_name,
        )?;
    }

    for (ri, cr) in &crowdedness_list {
        let id = cf.residue_id(*ri);
        writeln!(
            out,
            "crowdedness\t{},{}\t{:.6}\t{}",
            id.chain_id,
            format_resnum(id.res_id, id.insertion_code),
            cr,
            backbone.bb[ri.0 as usize].res_name,
        )?;
    }

    for (ri, fr) in &freedom_list {
        let id = cf.residue_id(*ri);
        writeln!(
            out,
            "freedom\t{},{}\t{:.6}\t{}",
            id.chain_id,
            format_resnum(id.res_id, id.insertion_code),
            fr,
            backbone.bb[ri.0 as usize].res_name,
        )?;
    }

    for (&(ri, rj), &val) in interference_list
        .pairs
        .iter()
        .zip(&interference_list.degrees)
    {
        let id_a = cf.residue_id(ri);
        let id_b = cf.residue_id(rj);
        writeln!(
            out,
            "interference\t{},{}\t{},{}\t{:.6}\t{}\t{}",
            id_a.chain_id,
            format_resnum(id_a.res_id, id_a.insertion_code),
            id_b.chain_id,
            format_resnum(id_b.res_id, id_b.insertion_code),
            val,
            backbone.bb[ri.0 as usize].res_name,
            backbone.bb[rj.0 as usize].res_name,
        )?;
    }

    // SEQUENCE line.
    let seq: Vec<&str> = backbone.bb.iter().map(|rb| rb.res_name.as_str()).collect();
    writeln!(out, "SEQUENCE: {}", seq.join(" "))?;

    out.flush()?;
    Ok(())
}

fn format_resnum(res_id: i32, ins: char) -> String {
    if ins == ' ' {
        res_id.to_string()
    } else {
        format!("{}{}", res_id, ins)
    }
}

/// Parse a residue selection: comma/whitespace separated tokens `chain:start-end`,
/// `chain:number` or `chain`. If `spec` names an existing file its contents are used
/// (lines starting with `#` are ignored). Malformed tokens and tokens matching no residue
/// are errors.
fn parse_selection(spec: &str, backbone: &ProteinBackbone) -> Result<Vec<ResidueIndex>, String> {
    let text = if std::path::Path::new(spec).is_file() {
        fs::read_to_string(spec)
            .map_err(|e| format!("cannot read selection file '{spec}': {e}"))?
            .lines()
            .filter(|l| !l.trim_start().starts_with('#'))
            .collect::<Vec<_>>()
            .join(",")
    } else {
        spec.to_string()
    };

    let mut result = Vec::new();
    for part in text.split(|c: char| c == ',' || c.is_whitespace()) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let before = result.len();
        if let Some((chain, range)) = part.split_once(':') {
            let bad = || format!("malformed residue token '{part}' (expected chain:start-end)");
            let (start, end) = if let Some((s, e)) = range.split_once('-') {
                (
                    s.parse::<i32>().map_err(|_| bad())?,
                    e.parse::<i32>().map_err(|_| bad())?,
                )
            } else {
                let v = range.parse::<i32>().map_err(|_| bad())?;
                (v, v)
            };
            for (i, id) in backbone.ids.iter().enumerate() {
                if id.chain_id == chain && id.res_id >= start && id.res_id <= end {
                    result.push(ResidueIndex(i as u32));
                }
            }
        } else {
            for (i, id) in backbone.ids.iter().enumerate() {
                if id.chain_id == part {
                    result.push(ResidueIndex(i as u32));
                }
            }
        }
        if result.len() == before {
            return Err(format!("residue token '{part}' matches no residue"));
        }
    }
    if result.is_empty() {
        return Err("selection is empty (expected at least one residue token)".to_string());
    }
    result.sort_unstable();
    result.dedup();
    Ok(result)
}

/// Render an error with the option flag(s) responsible.
fn describe_error(e: &ConFindError) -> String {
    match e {
        ConFindError::InvalidConfig(c) => describe_config_error(c),
        other => other.to_string(),
    }
}

fn describe_config_error(c: &ConfigError) -> String {
    match c {
        ConfigError::InvalidCutoff { name, .. } => {
            let flag = match *name {
                "d_clash" => "--dclash",
                "d_cont" => "--dcont",
                "d_cut" => "--dcut",
                "tau_lo" => "--taulo",
                "tau_hi" => "--tauhi",
                _ => "--?",
            };
            format!("{flag}: {c}; allowed: a finite number > 0")
        }
        ConfigError::TauOrder { .. } => {
            format!("--taulo/--tauhi: {c}; allowed: --taulo < --tauhi (or --freedommode single-threshold)")
        }
        ConfigError::DcutTooSmall { .. } => {
            format!("--dcut: {c}; allowed: --dcut >= 2*max R_max + --dcont")
        }
        ConfigError::EmptyScope => {
            format!("--scope: {c}; allowed: 'all' or a non-empty residue list")
        }
        ConfigError::ScopeOutOfRange { .. } => format!("--scope: {c}"),
        ConfigError::UnknownVariant { option, .. } => format!("--{option}: {c}"),
    }
}

enum ScopeArg {
    All,
    Spec(String),
}

struct Opts {
    pdb: PathBuf,
    rlib: PathBuf,
    out: Option<PathBuf>,
    sel: Option<String>,
    cd_cut: f64,
    in_cut: f64,
    scope: ScopeArg,
    config: ConFindConfig,
}

fn parse_args(args: &[String]) -> Result<Opts, Box<dyn std::error::Error>> {
    let mut pdb = None;
    let mut rlib = None;
    let mut out = None;
    let mut sel = None;
    let mut cd_cut = 0.05;
    let mut in_cut = 0.0;
    let mut threads: Option<NonZeroUsize> = None;
    let mut scope = ScopeArg::All;
    let mut cutoffs = Cutoffs::default();
    let mut cb_clash = CbClashPolicy::default();
    let mut crowd = CrowdednessMode::default();
    let mut freedom = FreedomMode::default();

    fn number(flag: &str, value: &str) -> Result<f64, String> {
        value
            .parse::<f64>()
            .map_err(|_| format!("{flag}: invalid value '{value}'; allowed: a finite number"))
    }

    let mut i = 1;
    while i < args.len() {
        let flag = args[i].as_str();
        let mut value = || -> Result<&str, String> {
            i += 1;
            args.get(i)
                .map(String::as_str)
                .ok_or_else(|| format!("{flag}: missing value"))
        };
        match flag {
            "--p" => pdb = Some(PathBuf::from(value()?)),
            "--rLib" => rlib = Some(PathBuf::from(value()?)),
            "--o" => out = Some(PathBuf::from(value()?)),
            "--sel" => sel = Some(value()?.to_string()),
            "--cdcut" => {
                let raw = value()?;
                let v = number(flag, raw)?;
                if !v.is_finite() {
                    return Err(format!("{flag}: invalid value '{raw}'; allowed: a finite number").into());
                }
                cd_cut = v;
            }
            "--incut" => {
                let raw = value()?;
                let v = number(flag, raw)?;
                if !v.is_finite() {
                    return Err(format!("{flag}: invalid value '{raw}'; allowed: a finite number").into());
                }
                in_cut = v;
            }
            "--threads" => {
                let v = value()?;
                let n = v
                    .parse::<usize>()
                    .ok()
                    .and_then(NonZeroUsize::new)
                    .ok_or_else(|| {
                        format!(
                            "--threads: invalid value '{v}'; allowed: a positive integer (>= 1)"
                        )
                    })?;
                threads = Some(n);
            }
            "--cbclash" => {
                cb_clash = value()?
                    .parse::<CbClashPolicy>()
                    .map_err(|e| describe_config_error(&e))?
            }
            "--crowdmode" => {
                crowd = value()?
                    .parse::<CrowdednessMode>()
                    .map_err(|e| describe_config_error(&e))?
            }
            "--freedommode" => {
                freedom = value()?
                    .parse::<FreedomMode>()
                    .map_err(|e| describe_config_error(&e))?
            }
            "--scope" => {
                let v = value()?;
                scope = if v == "all" {
                    ScopeArg::All
                } else {
                    ScopeArg::Spec(v.to_string())
                };
            }
            "--dclash" => cutoffs.d_clash = number(flag, value()?)?,
            "--dcont" => cutoffs.d_cont = number(flag, value()?)?,
            "--dcut" => cutoffs.d_cut = number(flag, value()?)?,
            "--taulo" => cutoffs.tau_lo = number(flag, value()?)?,
            "--tauhi" => cutoffs.tau_hi = number(flag, value()?)?,
            other => return Err(format!("unknown option '{other}'").into()),
        }
        i += 1;
    }

    let config = ConFindConfig::default()
        .with_cutoffs(cutoffs)
        .with_cb_clash_policy(cb_clash)
        .with_crowdedness_mode(crowd)
        .with_freedom_mode(freedom)
        .with_threads(threads);
    config.validate().map_err(|e| describe_config_error(&e))?;

    Ok(Opts {
        pdb: pdb.ok_or("--p <pdb> required")?,
        rlib: rlib.ok_or("--rLib <path> required")?,
        out,
        sel,
        cd_cut,
        in_cut,
        scope,
        config,
    })
}
