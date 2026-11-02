"""Sequence Quality Control (seqqc) submodule for proxide.

Provides pure-Python/NumPy utilities for sequence composition analysis,
net charge titration (heuristic and Bjellqvist), isoelectric point, hydropathy (GRAVY),
aromatic fraction, sequence entropy, and screening against background composition.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

AA: str = "ACDEFGHIKLMNPQRSTVWY"

# UniProt / SwissProt Release 2021 canonical background composition
SWISSPROT_BG: dict[str, float] = {
  "A": 0.0825,
  "C": 0.0137,
  "D": 0.0545,
  "E": 0.0675,
  "F": 0.0386,
  "G": 0.0707,
  "H": 0.0227,
  "I": 0.0596,
  "K": 0.0584,
  "L": 0.0966,
  "M": 0.0242,
  "N": 0.0406,
  "P": 0.0470,
  "Q": 0.0393,
  "R": 0.0553,
  "S": 0.0656,
  "T": 0.0534,
  "V": 0.0687,
  "W": 0.0108,
  "Y": 0.0292,
}

# Kyte-Doolittle hydropathy scale
KYTE_DOOLITTLE: dict[str, float] = {
  "A": 1.8,
  "C": 2.5,
  "D": -3.5,
  "E": -3.5,
  "F": 2.8,
  "G": -0.4,
  "H": -3.2,
  "I": 4.5,
  "K": -3.9,
  "L": 3.8,
  "M": 1.9,
  "N": -3.5,
  "P": -1.6,
  "Q": -3.5,
  "R": -4.5,
  "S": -0.8,
  "T": -0.7,
  "V": 4.2,
  "W": -0.9,
  "Y": -1.3,
}

# Bjellqvist pKa values
BJELLQVIST_PKA: dict[str, float] = {
  "N_term": 7.50,
  "C_term": 3.55,
  "K": 10.00,
  "R": 12.00,
  "H": 5.98,
  "D": 4.05,
  "E": 4.45,
  "C": 9.00,
  "Y": 10.00,
}


def _validate_sequence(sequence: str) -> str:
  """Validate sequence non-empty and canonical amino acids, normalized to upper case."""
  if not isinstance(sequence, str):
    raise TypeError(f"Expected str sequence, got {type(sequence).__name__}")
  if len(sequence) == 0:
    raise ValueError("Sequence cannot be empty")
  seq = sequence.upper()
  for c in seq:
    if c not in AA:
      raise ValueError(f"Invalid amino acid character: '{c}'")
  return seq


def amino_acid_frequencies(sequence: str) -> dict[str, float]:
  """Calculate normalized amino acid frequencies for each of the 20 canonical amino acids."""
  seq = _validate_sequence(sequence)
  length = len(seq)
  return {a: float(seq.count(a) / length) for a in AA}


def net_charge(sequence: str, ph: float = 7.4, model: str = "heuristic") -> float:
  """Calculate protein net charge under heuristic or Bjellqvist Henderson-Hasselbalch models.

  - model == 'heuristic' and ph == 7.4:
    q = count(K) + count(R) + 0.1 * count(H) - count(D) - count(E)
  - model == 'bjellqvist':
    Henderson-Hasselbalch titration with Bjellqvist pKa values.
  """
  seq = _validate_sequence(sequence)
  if model == "heuristic":
    if not np.isclose(ph, 7.4):
      raise ValueError(f"Heuristic net_charge model is only defined at pH 7.4, got {ph}")
    return float(
      seq.count("K") + seq.count("R") + 0.1 * seq.count("H") - seq.count("D") - seq.count("E")
    )
  elif model == "bjellqvist":
    # Basic groups (positive when protonated: 1 / (1 + 10^(pH - pKa)))
    q = 1.0 / (1.0 + 10.0 ** (ph - BJELLQVIST_PKA["N_term"]))
    q += seq.count("K") * (1.0 / (1.0 + 10.0 ** (ph - BJELLQVIST_PKA["K"])))
    q += seq.count("R") * (1.0 / (1.0 + 10.0 ** (ph - BJELLQVIST_PKA["R"])))
    q += seq.count("H") * (1.0 / (1.0 + 10.0 ** (ph - BJELLQVIST_PKA["H"])))

    # Acidic groups (negative when deprotonated: -1 / (1 + 10^(pKa - pH)))
    q -= 1.0 / (1.0 + 10.0 ** (BJELLQVIST_PKA["C_term"] - ph))
    q -= seq.count("D") * (1.0 / (1.0 + 10.0 ** (BJELLQVIST_PKA["D"] - ph)))
    q -= seq.count("E") * (1.0 / (1.0 + 10.0 ** (BJELLQVIST_PKA["E"] - ph)))
    q -= seq.count("C") * (1.0 / (1.0 + 10.0 ** (BJELLQVIST_PKA["C"] - ph)))
    q -= seq.count("Y") * (1.0 / (1.0 + 10.0 ** (BJELLQVIST_PKA["Y"] - ph)))
    return float(q)
  else:
    raise ValueError(f"Unknown net_charge model: '{model}'. Expected 'heuristic' or 'bjellqvist'.")


def isoelectric_point(sequence: str, model: str = "bjellqvist") -> float:
  """Find isoelectric point (pI) where net charge is 0.0 via bisection within 1e-4 tolerance."""
  seq = _validate_sequence(sequence)
  if model != "bjellqvist":
    raise ValueError(f"Unsupported model for isoelectric point: '{model}'. Expected 'bjellqvist'.")

  low, high = 0.0, 14.0
  while (high - low) > 1e-4:
    mid = (low + high) / 2.0
    q = net_charge(seq, ph=mid, model=model)
    if q > 0.0:
      low = mid
    else:
      high = mid
  return round((low + high) / 2.0, 4)


def gravy(sequence: str) -> float:
  """Calculate Grand Average of Hydropathy (GRAVY) using the Kyte-Doolittle scale."""
  seq = _validate_sequence(sequence)
  return float(sum(KYTE_DOOLITTLE[a] for a in seq) / len(seq))


def aromatic_fraction(sequence: str) -> float:
  """Calculate fraction of aromatic residues (F, W, Y)."""
  seq = _validate_sequence(sequence)
  return float((seq.count("F") + seq.count("W") + seq.count("Y")) / len(seq))


def sequence_entropy(sequence: str, base: float = 2.0, normalized: bool = False) -> float:
  """Calculate Shannon sequence entropy H = -sum(f_a * log_base(f_a))."""
  seq = _validate_sequence(sequence)
  length = len(seq)
  unique_counts = [seq.count(a) for a in set(seq)]
  h = 0.0
  log_base = math.log(base)
  for count in unique_counts:
    if count > 0:
      p = count / length
      h -= p * (math.log(p) / log_base)
  if normalized:
    h_max = math.log(20) / log_base
    h = h / h_max
  return float(h)


def screen(
  sequences: list[str] | dict[str, list[str]] | np.ndarray,
  background: dict[str, float] | None = None,
  threshold_ek: float = 0.30,
  pad_mask: np.ndarray | None = None,
) -> dict[str, Any]:
  """Screen sequence list, group dict, or padded 2D array against background composition."""
  bg = background or SWISSPROT_BG

  # Internal controls check (selftest gate)
  ctl_pos = (
    bool(np.isclose(amino_acid_frequencies("EEEE")["E"], 1.0))
    and abs(sum(bg.values()) - 1.0) < 0.01
  )
  ctl_neg = not np.isclose(amino_acid_frequencies("EEKK")["E"], 1.0)
  controls_pass = bool(ctl_pos and ctl_neg)

  # Coerce input sequences to dictionary of groups
  if isinstance(sequences, np.ndarray):
    if pad_mask is None:
      raise ValueError("pad_mask must be provided for 2D array representation")
    if pad_mask.shape != sequences.shape:
      raise ValueError(
        f"pad_mask shape {pad_mask.shape} does not match sequences shape {sequences.shape}"
      )
    seqs_list: list[str] = []
    for i in range(sequences.shape[0]):
      valid_tokens = sequences[i][pad_mask[i]]
      if np.issubdtype(sequences.dtype, np.integer):
        seq_str = "".join(AA[int(tok)] for tok in valid_tokens)
      else:
        seq_str = "".join(str(tok) for tok in valid_tokens)
      _validate_sequence(seq_str)
      seqs_list.append(seq_str)
    groups = {"all": seqs_list}
    is_dict_input = False
  elif isinstance(sequences, list):
    if not sequences:
      raise ValueError("Sequence list cannot be empty")
    validated_seqs = [_validate_sequence(s) for s in sequences]
    groups = {"all": validated_seqs}
    is_dict_input = False
  elif isinstance(sequences, dict):
    if not sequences:
      raise ValueError("Sequence dictionary cannot be empty")
    groups = {}
    total_count = 0
    for grp_name, grp_seqs in sequences.items():
      if grp_seqs:
        validated = [_validate_sequence(s) for s in grp_seqs]
        groups[grp_name] = validated
        total_count += len(validated)
    if total_count == 0:
      raise ValueError("All sequence groups are empty")
    is_dict_input = True
  else:
    raise TypeError(f"Unsupported sequences type: {type(sequences).__name__}")

  # Pool all sequences across groups
  all_seqs: list[str] = [s for grp in groups.values() for s in grp]
  n_seqs = len(all_seqs)

  pool_eks = [(s.count("E") + s.count("K")) / len(s) for s in all_seqs]
  pool_charges = [net_charge(s, ph=7.4, model="heuristic") for s in all_seqs]
  pool_gravies = [gravy(s) for s in all_seqs]
  pool_aromatics = [aromatic_fraction(s) for s in all_seqs]
  pool_entropies = [sequence_entropy(s) for s in all_seqs]

  mean_ek = float(np.mean(pool_eks))
  frac_ek_gt_30pct = float(np.mean([ek > threshold_ek for ek in pool_eks]))
  mean_net_charge = float(np.mean(pool_charges))
  frac_net_neg = float(np.mean([c < 0.0 for c in pool_charges]))
  mean_gravy = float(np.mean(pool_gravies))
  mean_aromatic = float(np.mean(pool_aromatics))
  mean_entropy = float(np.mean(pool_entropies))

  bg_ek = float(bg["E"] + bg["K"])
  pool_ek_enrichment = float(mean_ek / bg_ek)

  pool_mean_freq = {a: float(np.mean([s.count(a) / len(s) for s in all_seqs])) for a in AA}
  pool_fold = {a: float(pool_mean_freq[a] / bg[a]) for a in AA}

  # Screening design gates
  flag_reasons: list[str] = []
  if mean_ek > threshold_ek:
    flag_reasons.append("exceeds_ek_threshold")
  if mean_aromatic < 0.02:
    flag_reasons.append("low_aromatic_fraction")
  screen_pass = len(flag_reasons) == 0

  # Summarize per group
  groups_out: dict[str, dict[str, Any]] = {}
  for grp_name, grp_seqs in groups.items():
    g_len = [len(s) for s in grp_seqs]
    g_eks = [(s.count("E") + s.count("K")) / len(s) for s in grp_seqs]
    g_charges = [net_charge(s, ph=7.4, model="heuristic") for s in grp_seqs]
    g_freqs = {a: round(float(np.mean([s.count(a) / len(s) for s in grp_seqs])), 4) for a in AA}
    groups_out[grp_name] = {
      "n": len(grp_seqs),
      "mean_len": float(np.mean(g_len)),
      "mean_freq": g_freqs,
      "mean_ek": round(float(np.mean(g_eks)), 4),
      "frac_ek_gt_30pct": round(float(np.mean([ek > threshold_ek for ek in g_eks])), 3),
      "mean_net_charge": round(float(np.mean(g_charges)), 2),
      "frac_net_neg": round(float(np.mean([c < 0.0 for c in g_charges])), 3),
    }

  if is_dict_input:
    fold_over_background: Any = {
      grp_name: {a: round(groups_out[grp_name]["mean_freq"][a] / bg[a], 2) for a in AA}
      for grp_name in groups_out
    }
  else:
    fold_over_background = {a: round(pool_fold[a], 2) for a in AA}

  res: dict[str, Any] = {
    "controls_pass": controls_pass,
    "screen_pass": screen_pass,
    "flag_reasons": flag_reasons,
    "n_sequences": n_seqs,
    "background_ek": round(bg_ek, 4),
    "mean_ek": mean_ek,
    "frac_ek_gt_30pct": frac_ek_gt_30pct,
    "mean_net_charge": mean_net_charge,
    "frac_net_neg": frac_net_neg,
    "mean_gravy": mean_gravy,
    "mean_aromatic_fraction": mean_aromatic,
    "mean_entropy": mean_entropy,
    "pool_ek_enrichment": round(pool_ek_enrichment, 2),
    "mean_freq": {a: round(pool_mean_freq[a], 4) for a in AA},
    "fold_over_background": fold_over_background,
    "groups": groups_out,
  }

  if "pool_all" in groups_out:
    res["pool_mean_ek"] = groups_out["pool_all"]["mean_ek"]
    res["pool_n"] = groups_out["pool_all"]["n"]
  if "final_five" in groups_out:
    res["final_mean_ek"] = groups_out["final_five"]["mean_ek"]

  return res


__all__ = [
  "AA",
  "SWISSPROT_BG",
  "KYTE_DOOLITTLE",
  "BJELLQVIST_PKA",
  "amino_acid_frequencies",
  "net_charge",
  "isoelectric_point",
  "gravy",
  "aromatic_fraction",
  "sequence_entropy",
  "screen",
]
