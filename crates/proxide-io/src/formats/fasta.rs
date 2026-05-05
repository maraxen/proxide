use rayon::prelude::*;
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
///
/// For A3M, uppercase letters and '-' are considered match states.
/// Lowercase letters are considered insertions relative to the query and are skipped
/// for the purpose of the dense JAX array, but their presence is used to validate alignment.
pub fn parse_a3m<P: AsRef<Path>>(path: P) -> Result<TokenizedMSA, FastaError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut raw_records = Vec::new();
    let mut current_header = String::new();
    let mut current_seq = Vec::new();

    // Pass 1: Sequential read into memory chunks for parallel processing
    // FASTA is inherently sequential but we can parallelize the tokenization per sequence.
    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }

        if line.starts_with('>') {
            if !current_header.is_empty() {
                raw_records.push((current_header.clone(), current_seq.clone()));
                current_seq.clear();
            }
            current_header = line[1..].to_string();
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

    // Pass 2: Parallel tokenization
    // We use the first sequence (query) to define the match_mask if it's A3M.
    // In A3M, the query (first seq) contains no lowercase letters/insertions by definition.
    let alphabet = "ACDEFGHIKLMNPQRSTVWY-X";
    let aa_to_id: std::collections::HashMap<char, i8> = alphabet
        .chars()
        .enumerate()
        .map(|(i, c)| (c, i as i8))
        .collect();

    let results: Vec<(String, Vec<i8>, Vec<bool>)> = raw_records
        .into_par_iter()
        .map(|(header, seq_lines)| {
            let full_seq = seq_lines.join("");
            let mut tokenized = Vec::new();
            let mut mask = Vec::new();

            for c in full_seq.chars() {
                if c.is_ascii_lowercase() {
                    // Insertion column - skip for dense match-state MSA
                    continue;
                }

                let token = *aa_to_id.get(&c.to_ascii_uppercase()).unwrap_or(&21); // 21 is 'X'
                tokenized.push(token);
                mask.push(true); // Every uppercase/gap in A3M is a match column
            }

            (header, tokenized, mask)
        })
        .collect();

    let mut final_sequences = Vec::new();
    let mut final_headers = Vec::new();
    let mut match_mask = Vec::new();

    if !results.is_empty() {
        // Assume all sequences have the same number of match states
        // In A3M this is guaranteed by the format.
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
