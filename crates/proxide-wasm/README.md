# proxide-wasm

WASM-compilable GAFF2 bonded-parameter resolver for use in browser-based molecular dynamics demos.

## Quick Start (Browser)

Add the compiled WASM module to your HTML and use the JavaScript API:

```html
<script type="module">
  import init, { smiles_to_params, version } from "./pkg/proxide_wasm.js";

  async function run() {
    await init();
    
    const smiles = "CCO";  // ethanol
    const json = smiles_to_params(smiles);
    const params = JSON.parse(json);
    
    console.log(`proxide-wasm v${version()}`);
    console.log(params);
  }

  run();
</script>
```

See `demo/index.html` for a complete example with interactive SMILES input and parameter display.

## Building

### Build the WASM module

```bash
# Install wasm-pack if you haven't already
cargo install wasm-pack

# Build to web target
cd crates/proxide-wasm
wasm-pack build --target web --out-dir pkg
```

The `pkg/` directory now contains `proxide_wasm.js` and `proxide_wasm_bg.wasm` (~991 KB).

### Run the demo

```bash
# From the proxide-wasm directory
python3 -m http.server 8080
```

Then open `http://localhost:8080/demo/` in your browser.

## JavaScript API

### `parse_smiles(smiles: string) → object | Error`

Parses a SMILES string and returns a molecular graph with atoms and bonds.

```javascript
const mol = parse_smiles("c1ccccc1");  // benzene
console.log(mol.atoms);  // [{ idx: 0, atomic_num: 6, atom_type: "ca" }, ...]
```

### `parse_pdb(text: string) → object | Error`

Parses PDB file text and returns a molecular graph. **Note:** Bond connectivity is only populated when the PDB file includes `CONECT` records. Standard protein PDB files omit `CONECT`, so `mol.bonds` will be empty — calling `assign_params_js` on such a molecule will produce an empty parameter set. Full peptide-bond inference is deferred to v1.0 (see Known Limitations).

```javascript
const mol = parse_pdb(pdbText);  // bonds[] empty unless CONECT records present
```

### `assign_params_js(mol: object) → string | Error`

Assigns GAFF2 bonded parameters to a previously parsed molecule. Returns JSON string.

```javascript
const mol = parse_smiles("CC");
const jsonParams = assign_params_js(mol);
const params = JSON.parse(jsonParams);
```

### `smiles_to_params(smiles: string) → string | Error`

Combined pipeline: parse SMILES + assign GAFF2 parameters, return JSON string.

```javascript
const json = smiles_to_params("CCO");
const params = JSON.parse(json);
console.log(params.bonds[0]);  // { i: 0, j: 1, k: 194556.0, r0: 0.1538, type_pair: ["c3", "c3"] }
```

### `version() → string`

Returns the proxide-wasm crate version.

```javascript
console.log(version());  // "0.1.0"
```

## JSON Output Schema

The parameter assignment returns a JSON object conforming to the prolix §7.1 contract:

```json
{
  "molecule_id": "mol",
  "smiles": null,
  "atoms": [
    {
      "idx": 0,
      "atomic_num": 6,
      "atom_type": "c3"
    }
  ],
  "bonds": [
    {
      "i": 0,
      "j": 1,
      "k": 194556.0,
      "r0": 0.1538,
      "type_pair": ["c3", "c3"]
    }
  ],
  "angles": [
    {
      "i": 0,
      "j": 1,
      "k_idx": 2,
      "k_angle": 527.0,
      "theta0": 1.911,
      "type_triple": ["c3", "c3", "c3"]
    }
  ],
  "torsions": [
    {
      "i": 0,
      "j": 1,
      "k_idx": 2,
      "l": 3,
      "periodicity": 3,
      "k_torsion": 0.6276,
      "phase": 0.0,
      "type_quad": ["c3", "c3", "c3", "c3"]
    }
  ]
}
```

### Units

- **k** (bond force constant): kJ/mol/nm²
- **r0** (equilibrium bond length): nm
- **angle k**: kJ/mol/rad²
- **θ₀** (equilibrium angle): radians
- **torsion pk**: kcal/mol per periodicity (will be converted by consumer)

These are OpenMM convention units. Non-bonded parameters (LJ ε, σ, partial charges) are not included in v0.

## Equivalence Test

Verify structural correctness and physical plausibility against the Python GAFF2 implementation:

```bash
# Build the Rust CLI binary
cargo build -p proxide-wasm --bin param_cli

# Run the test suite (requires Python dev dependencies)
uv run pytest tests/test_hp4_wasm_parity.py -v
```

The test suite (`tests/test_hp4_wasm_parity.py`) runs 15 tests across 5 molecules (methane, ethane, ethanol, acetone, benzene) to verify:
- Atom type assignment correctness
- Bond parameter physical ranges (e.g., C–C ~1.5 Å, k ~ 200 kJ/mol/nm²)
- Angle and torsion parameter consistency
- Parity with `openmmforcefields.GAFFTemplateGenerator` using the same GAFF2 forcefield file

Both sides read the same `gaff-2.11.xml` file, so differences only arise from atom-typing rules.

## Known Limitations (v0)

- **Element scope**: H, C, N, O only (ANI-1x / GAFF2 v0 scope). Sulfur, phosphorus, and halogens are not typed and will cause errors.

- **PDB bond connectivity**: `parse_pdb` only populates bond connectivity when the PDB file includes explicit `CONECT` records. Standard protein PDB files omit these (bonds are implied by residue topology), so calling `assign_params_js` on a PDB-parsed molecule without `CONECT` records will produce an empty parameter set. Peptide-bond inference from residue topology is deferred to v1.0.

- **Fused polycyclic rings**: Ring detection now correctly identifies rings of up to size 8 using iterative DFS. Multi-ring systems (naphthalene, indole) are handled correctly — each ring atom is marked via cycle-path tracing from back edges.

- **No non-bonded parameters**: LJ epsilon/sigma and partial charges (AM1-BCC, RESP, etc.) are out of scope for v0. The JSON output contains bonded parameters only.

- **SMIRKS vs. rules-based typing**: This crate uses a rules-based GAFF2 atom typer, not a full SMIRKS pattern engine. Unusual chemistries not covered by the rules may fail or produce incorrect types.

- **WASM binary size**: `wasm-opt` is disabled due to a bulk-memory validation issue in the current `wasm-pack` release. The unoptimized binary is 991 KB. This will be reduced in v1.0 when `wasm-pack` fixes the validator.

## Cross-References

- **Prolix backlog item 349** (HP4-WASM): Delivers the WASM module for browser-based hydrogen-bonded quartet (HP4) ensemble calculations.
- **Prolix §7.1-WASM**: Sister deliverable in the prolix project. Once the HP4 20-molecule ensemble ships, this crate will be wired into the prolix browser demo for interactive parameter exploration.

## License

MIT
