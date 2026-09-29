# proxide-rotlib

> **Alpha — experimental, no API stability guarantees.** This crate is part of an active research project; APIs may change without notice between versions. Not recommended for production use.

Backbone-dependent rotamer library for protein sidechain placement.

## Data Provenance & Licensing

This crate includes **two distinct licenses**:

- **Code**: MIT-licensed (workspace LICENSE) — covers all Rust implementation in `src/`.
- **Rotamer Data**: **ODC-BY-1.0** — the rotamer coordinates and backbone-dependent statistics are derived from the Dunbrack 2010 backbone-dependent rotamer library and are made available under the Open Data Commons Attribution License 1.0.

### Attribution Notice

The rotamer data embedded in all generated protobuf artifacts carries this attribution (ODC-BY-1.0 requirement):

> Contains information from the 2010 Backbone-Dependent Rotamer Library (http://dunbrack.fccc.edu/bbdep2010), made available under the ODC Attribution License (http://dunbrack.fccc.edu/bbdep2010/license/bbdep2010_license.txt).

This attribution is recorded in the `RotamerLibrary.attribution` field of every compiled `.pb.zst` artifact. The protobuf loader enforces that this field is present and non-empty.

### Data Source Note

The MASTER/Mosaist `rotlib.bin` file (published under CC BY-NC-SA) is **not** redistributed by this crate. All rotamer coordinates are rebuilt from the Dunbrack ODC-BY text library using standard protein geometry, ensuring the ODC-BY license applies to the derived coordinates.

### Data Attribution

This crate ships pre-built rotamer libraries derived from:

- **Dunbrack BBDEP 2010 (SimpleOpt1-5)** — Shapovalov & Dunbrack (2011), *Structure* 19(6):844–858. License: **ODC-BY-1.0** (Open Data Commons Attribution). Attribution: "Contains data from the Dunbrack Backbone-Dependent Rotamer Library."
- **PDB Chemical Component Dictionary (CCD)** — RCSB PDB. License: **CC0-1.0**.

### Synthetic ALA entry (backlog #5244)

The Dunbrack BBDEP format only tabulates chi-bearing ("rotameric") residues, so it never
contains ALA (or GLY) — ALA has no chi angles. `data/rotlibs/proxide-rotlib-dunbrack2010-ccd.pb.zst`
therefore carries one entry that is **not** Dunbrack ODC-BY data: a synthesized ALA
`ResidueEntry`, added by `proxide_rotlib::add_synthetic_ala` (called from `build_library`
when `convert_rotlib` is run with `--synthesize-ala`). It consists of:

- One (phi, psi) bin, centered at the structural placeholder (0.0, 0.0) — unread by any
  query path (`find_closest_angle` against a single-element grid always returns index 0;
  `load_pb` builds a 1×1 grid).
- One rotamer at probability 1.0, `num_chi = 0`, `chi = []`.
- CB built through the exact same `standard_residue_template("ALA")` +
  `build_standard_sidechain` path (Engh–Huber placeholder geometry: 1.540 Å / 110.5° /
  −119.7°) used for every other non-proline residue in this build. **This CB bond length
  is documented as placeholder geometry of unverified provenance — it is NOT cited as
  Engh & Huber** (see the filed debt on backlog #5244 to verify the ~1.52 Å literature
  value before relying on it for anything precision-sensitive).

Verified against Mosaist's own `rotlib.bin` (`RotamerLibrary::load()`, aggregate
statistics only — no raw bytes read or copied): Mosaist's ALA is also 1 bin, 1 rotamer,
p=1.0, with 0.0 Å CB spread across bins, so this synthetic model is faithful to Mosaist's
own representation for ALA specifically.

This is recorded in the artifact's `provenance` field (e.g. `"...; ALA: synthetic, 1
rotamer, p=1, geometry=<source>"`) so it is traceable from the artifact alone, without
needing this README.

### ODC-BY-1.0 License

Full text: https://opendatacommons.org/licenses/by/1-0/

## Citation

If you use this library in research, please cite:

Shapovalov MV, Dunbrack RL Jr. "A smoothed backbone-dependent rotamer library for proteins derived from adaptive kernel density estimates and regressions." *Structure* 19(6):844–858 (2011).

## Usage

The crate supports two loader paths:

- **Protobuf path** (preferred): `RotamerLibrary::load_pb()` — loads precomputed rotamer coordinates from the Dunbrack BBDEP2010 protobuf artifact.
- **MSL binary path** (legacy): `RotamerLibrary::load()` — loads the MSL binary format for backward compatibility.

To regenerate the protobuf artifact from the Dunbrack text library, use the `convert_rotlib` binary.

### Regenerating `data/rotlibs/proxide-rotlib-dunbrack2010-ccd.pb.zst`

Recipe recovered 2026-09-23 (backlog #5244) by decoding the previously-committed artifact's
`provenance`/`geometry_source`/`geometry_license` fields (the latter two are empty strings,
meaning no `--ic-source` was passed — the "-ccd" in the filename is a legacy name from an
earlier converter design (`.claude/workflows/sprint14-rotlib-ccd-migration.js`) and does not
mean a CCD IC table is actually applied; geometry is the plain Engh–Huber template defaults)
and confirming the input `.lib`'s sha256 against `.praxia/docs/specs/260602_dunbrack-rotlib-protobuf-cis-pro.md:303`.
Rebuilding with this recipe and no `--synthesize-ala` reproduces the previously-committed
artifact's decoded contents field-for-field (allow-list: `provenance`/`attribution` text only —
see debt on the attribution wording drift, which predates this sprint and is unrelated to
ALA). Rebuilding with `--synthesize-ala` adds exactly one ALA entry and nothing else changes.

```bash
cargo run --release --bin convert_rotlib -- \
  --rotlib-source "dunbrack:/home/marielle/projects/proxide/data/rotlibs/SimpleOpt1-5/ALL.bbdep.rotamers.lib" \
  --output data/rotlibs/proxide-rotlib-dunbrack2010-ccd.pb.zst \
  --synthesize-ala
```

Input `.lib` sha256: `aade9d4fe6ede1bd669dc8a145fc86321e3342fe7c19d0da914bdcbc0ff34bdf`
(SimpleOpt1-5/ALL.bbdep.rotamers.lib, matches the sha recorded at
`260602_dunbrack-rotlib-protobuf-cis-pro.md:303`).

Regeneration history for this committed artifact:

| date | sha256 | size (bytes) | notes |
|---|---|---|---|
| (prior) | `13264f972b782970141032617e1395932714abece81d27980b5749e15268579b` | 12609399 | no ALA (pre-#5244) |
| 2026-09-23 | `a923b1c046a3bf361bb43c10c7e7a04cb4ddd650da1a971b0e7e1bac9fb4eaa9` | 12609284 | +1 synthetic ALA entry (#5244); every other residue's bins bit-for-bit unchanged (verified by full decoded-message diff, allow-listing only `provenance`/`attribution`) |
