# proxide-rotlib

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
