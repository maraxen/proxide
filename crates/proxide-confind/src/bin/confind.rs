use proxide_confind::{
    load_pdb_f64, ConFind, ResidueIndex,
};
use proxide_rotlib::RotamerLibrary;
use std::fs;
use std::io::{BufWriter, Write};
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

    // orx-parallel doesn't have a global thread pool configuration;
    // thread count is set per-call via .num_threads() on the par iterator.
    if let Some(_n) = opts.threads {
        log::info!("note: thread count is configured per-operation in orx-parallel");
    }

    let cf = ConFind::new(rotlib, backbone.clone(), false);

    // Determine residue selection.
    let residues: Vec<ResidueIndex> = if let Some(ref sel) = opts.sel {
        parse_selection(sel, &backbone)
    } else {
        (0..backbone.bb.len() as u32).map(ResidueIndex).collect()
    };

    log::info!("running contacts on {} residues...", residues.len());
    let contact_list = cf.contacts(&residues, opts.cd_cut)?;

    let crowdedness_list: Vec<(ResidueIndex, f64)> = residues
        .iter()
        .filter_map(|&ri| cf.crowdedness(ri).ok().map(|c| (ri, c)))
        .collect();

    let freedom_list: Vec<(ResidueIndex, f64)> = residues
        .iter()
        .filter_map(|&ri| cf.freedom(ri).ok().map(|f| (ri, f)))
        .collect();

    let interference_list = cf.interference(&residues, opts.in_cut)?;

    // Write output.
    let out: Box<dyn Write> = if let Some(ref path) = opts.out {
        Box::new(BufWriter::new(fs::File::create(path)?))
    } else {
        Box::new(BufWriter::new(std::io::stdout()))
    };
    let mut out = out;

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
            "crwdnes\t{},{}\t{:.6}\t{}",
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

    for (&(ri, rj), &val) in interference_list.pairs.iter().zip(&interference_list.degrees) {
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

    Ok(())
}

fn format_resnum(res_id: i32, ins: char) -> String {
    if ins == ' ' {
        res_id.to_string()
    } else {
        format!("{}{}", res_id, ins)
    }
}

fn parse_selection(sel: &str, backbone: &proxide_confind::ProteinBackbone) -> Vec<ResidueIndex> {
    // Format: chain:start-end (e.g. "A:10-50") or chain (e.g. "A").
    let mut result = Vec::new();
    for part in sel.split(',') {
        let part = part.trim();
        if let Some((chain, range)) = part.split_once(':') {
            let (start, end) = if let Some((s, e)) = range.split_once('-') {
                (s.parse::<i32>().unwrap_or(i32::MIN), e.parse::<i32>().unwrap_or(i32::MAX))
            } else {
                let v = range.parse::<i32>().unwrap_or(0);
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
    }
    result.sort_unstable();
    result.dedup();
    result
}

struct Opts {
    pdb: PathBuf,
    rlib: PathBuf,
    out: Option<PathBuf>,
    sel: Option<String>,
    cd_cut: f64,
    in_cut: f64,
    threads: Option<usize>,
}

fn parse_args(args: &[String]) -> Result<Opts, Box<dyn std::error::Error>> {
    let mut pdb = None;
    let mut rlib = None;
    let mut out = None;
    let mut sel = None;
    let mut cd_cut = 0.05;
    let mut in_cut = 0.0;
    let mut threads = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--p" => {
                i += 1;
                pdb = Some(PathBuf::from(&args[i]));
            }
            "--rLib" => {
                i += 1;
                rlib = Some(PathBuf::from(&args[i]));
            }
            "--o" => {
                i += 1;
                out = Some(PathBuf::from(&args[i]));
            }
            "--sel" => {
                i += 1;
                sel = Some(args[i].clone());
            }
            "--cdcut" => {
                i += 1;
                cd_cut = args[i].parse()?;
            }
            "--incut" => {
                i += 1;
                in_cut = args[i].parse()?;
            }
            "--threads" => {
                i += 1;
                threads = Some(args[i].parse()?);
            }
            _ => {}
        }
        i += 1;
    }

    Ok(Opts {
        pdb: pdb.ok_or("--p <pdb> required")?,
        rlib: rlib.ok_or("--rLib <path> required")?,
        out,
        sel,
        cd_cut,
        in_cut,
        threads,
    })
}
