use proxide_jaccard::{
    accessions, pairwise_containment, pairwise_jaccard_distance, write_distance_matrix, SketchStore,
};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Clone, Copy)]
enum Metric {
    Jaccard,
    Containment,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

struct Args {
    input: PathBuf,
    accessions: Option<PathBuf>,
    out: PathBuf,
    metric: Metric,
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let raw: Vec<String> = std::env::args().collect();
    let args = parse_args(&raw)?;

    log::info!("loading sketches from {}", args.input.display());
    let wanted = match &args.accessions {
        Some(path) => Some(accessions::read_accession_list(path)?),
        None => None,
    };
    let store = SketchStore::load_parquet(&args.input, wanted.as_deref())?;
    log::info!("loaded {} accessions", store.len());

    let mat = match args.metric {
        Metric::Jaccard => {
            log::info!(
                "computing {0}x{0} pairwise jaccard distance matrix",
                store.len()
            );
            pairwise_jaccard_distance(&store)
        }
        Metric::Containment => {
            log::info!("computing {0}x{0} pairwise containment matrix", store.len());
            pairwise_containment(&store)
        }
    };

    write_distance_matrix(&args.out, &mat, store.accessions())?;
    log::info!("wrote {}", args.out.display());

    Ok(())
}

fn parse_args(raw: &[String]) -> Result<Args, Box<dyn std::error::Error>> {
    let mut input = None;
    let mut accessions = None;
    let mut out = None;
    let mut metric = Metric::Jaccard;

    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--input" => input = Some(PathBuf::from(next_arg(raw, &mut i)?)),
            "--accessions" => accessions = Some(PathBuf::from(next_arg(raw, &mut i)?)),
            "--out" => out = Some(PathBuf::from(next_arg(raw, &mut i)?)),
            "--metric" => {
                let value = next_arg(raw, &mut i)?;
                metric = match value.as_str() {
                    "jaccard" => Metric::Jaccard,
                    "containment" => Metric::Containment,
                    other => {
                        return Err(format!(
                            "unknown --metric value: {other} (expected jaccard or containment)"
                        )
                        .into())
                    }
                };
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 1;
    }

    Ok(Args {
        input: input.ok_or("missing required --input <parquet path>")?,
        accessions,
        out: out.ok_or("missing required --out <npy path>")?,
        metric,
    })
}

fn next_arg(raw: &[String], i: &mut usize) -> Result<String, Box<dyn std::error::Error>> {
    let flag = raw[*i].clone();
    *i += 1;
    raw.get(*i)
        .cloned()
        .ok_or_else(|| format!("missing value for {flag}").into())
}

fn print_usage() {
    eprintln!(
        "proxide-jaccard --input <minhash.parquet> --out <dist.npy> [--accessions <list.txt>] [--metric jaccard|containment]"
    );
    eprintln!();
    eprintln!("  --input        2-column parquet: accession (Utf8), hashes_list (List<Int64>)");
    eprintln!("  --out          output .npy path; a sidecar <stem>.accessions.txt is written alongside it");
    eprintln!("  --accessions   optional file, one accession per line; restricts the matrix to these rows");
    eprintln!("                 (omit to use every row in the parquet)");
    eprintln!("  --metric       jaccard (default, symmetric) or containment (asymmetric:");
    eprintln!("                 mat[i][j] = fraction of accession i contained in accession j)");
}
