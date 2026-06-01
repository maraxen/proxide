#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FastaError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
}

pub struct TokenizedMSA {
    pub sequences: Vec<Vec<i8>>,
    pub headers: Vec<String>,
    pub match_mask: Vec<bool>,
}

/// Parse A3M/FASTA in parallel using Rayon.
pub fn parse_a3m<P: AsRef<Path>>(path: P) -> Result<TokenizedMSA, FastaError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut raw_records = Vec::new();
    let mut current_header = String::new();
    let mut current_seq = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }

        if let Some(header_content) = line.strip_prefix('>') {
            if !current_header.is_empty() {
                raw_records.push((current_header.clone(), current_seq.clone()));
                current_seq.clear();
            }
            current_header = header_content.to_string();
        } else {
            current_seq.push(line);
        }
    }

    if !current_header.is_empty() {
        raw_records.push((current_header, current_seq));
    }

    if raw_records.is_empty() {
        return Err(FastaError::InvalidFormat("Empty file".into()));
    }

    let alphabet = "ACDEFGHIKLMNPQRSTVWY-X";
    let aa_to_id: HashMap<char, i8> = alphabet
        .chars()
        .enumerate()
        .map(|(i, c)| (c, i as i8))
        .collect();

    #[cfg(feature = "parallel")]
    let results: Vec<(String, Vec<i8>, Vec<bool>)> = raw_records
        .into_par_iter()
        .map(|(header, seq_lines)| tokenize_record(header, seq_lines, &aa_to_id))
        .collect();

    #[cfg(not(feature = "parallel"))]
    let results: Vec<(String, Vec<i8>, Vec<bool>)> = raw_records
        .into_iter()
        .map(|(header, seq_lines)| tokenize_record(header, seq_lines, &aa_to_id))
        .collect();

    let mut final_sequences = Vec::new();
    let mut final_headers = Vec::new();
    let mut match_mask = Vec::new();

    if !results.is_empty() {
        match_mask = results[0].2.clone();
        for (header, seq, _) in results {
            final_headers.push(header);
            final_sequences.push(seq);
        }
    }

    Ok(TokenizedMSA {
        sequences: final_sequences,
        headers: final_headers,
        match_mask,
    })
}

fn tokenize_record(header: String, seq_lines: Vec<String>, aa_to_id: &HashMap<char, i8>) -> (String, Vec<i8>, Vec<bool>) {
    let full_seq = seq_lines.join("");
    let mut tokenized = Vec::new();
    let mut mask = Vec::new();

    for c in full_seq.chars() {
        if c.is_ascii_lowercase() {
            continue;
        }

        let token = *aa_to_id.get(&c.to_ascii_uppercase()).unwrap_or(&21);
        tokenized.push(token);
        mask.push(true);
    }
    (header, tokenized, mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::fs::File;
    use std::env;

    fn temp_file(name: &str) -> std::path::PathBuf {
        let mut path = env::temp_dir();
        path.push(name);
        path
    }

    #[test]
    fn test_parse_fasta_simple() {
        let path = temp_file("test_fasta_simple.fasta");
        let mut file = File::create(&path).unwrap();
        writeln!(file, ">seq1\nACDEF").unwrap();
        writeln!(file, ">seq2\nGHIKL").unwrap();

        let msa = parse_a3m(&path).unwrap();
        assert_eq!(msa.headers.len(), 2);
        assert_eq!(msa.headers[0], "seq1");
        // A=0, C=1, D=2, E=3, F=4
        assert_eq!(msa.sequences[0], vec![0, 1, 2, 3, 4]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_parse_multiline() {
        let path = temp_file("test_multiline.fasta");
        let mut file = File::create(&path).unwrap();
        writeln!(file, ">seq1\nACD\nEF").unwrap();
        
        let msa = parse_a3m(&path).unwrap();
        assert_eq!(msa.sequences[0], vec![0, 1, 2, 3, 4]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_parse_a3m_insertions() {
        let path = temp_file("test_a3m.fasta");
        let mut file = File::create(&path).unwrap();
        writeln!(file, ">seq1\nACaDEF").unwrap();
        
        let msa = parse_a3m(&path).unwrap();
        assert_eq!(msa.sequences[0].len(), 5);
        assert_eq!(msa.sequences[0], vec![0, 1, 2, 3, 4]);
        std::fs::remove_file(path).unwrap();
    }
}
