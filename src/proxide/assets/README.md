# Force Field Assets

Only `gaff/` is vendored in this repository. Everything else is either
fetched on first use, or must be supplied by you — see
`.praxia/docs/specs/260824_license-provenance-verification.md` for the full
rationale.

## Fetch-on-demand (`amber/`, `water/`, `implicit/`, most of `openmm_bundled/`)

No licensing restriction found on audit, but these are fetched at runtime
from a pinned commit of [openmmforcefields](https://github.com/openmm/openmmforcefields)
or [OpenMM](https://github.com/openmm/openmm) and cached locally
(`proxide.assets._fetch`) rather than vendored — no library-fetching-at-build-time
staleness, and it keeps the wheel small. Calling `load_force_field("ff14SB")`
(or any other name in `proxide.assets._asset_index.ASSET_INDEX`) fetches and
caches transparently; nothing to configure. Cache location:
`platformdirs.user_cache_dir("proxide")/assets`, overridable via
`PROXIDE_ASSET_CACHE_DIR`.

## Bring-your-own (`charmm/`, plus CGenFF/CHARMM36-derived files bundled by OpenMM itself)

proxide never fetches or bundles CGenFF or the base CHARMM36 toppar files.
CGenFF requires a signed not-for-profit license agreement from the
MacKerell lab (mackerell.umaryland.edu/charmm_ff.shtml) — commercial use
goes through SilcsBio, a company the same lab operates. No positive
redistribution grant was found for the base CHARMM36 toppar files either,
and openmmforcefields' own distribution of the same converted files doesn't
appear to be independently documented as authorized — so proxide doesn't
treat "openmmforcefields does it too" as a resolved basis for bundling.

If you have your own rights to this content, set `PROXIDE_CHARMM_TOPPAR_DIR`
to a directory containing your copy of the converted OpenMM-XML toppar
files, and `load_force_field("charmm36_protein")` (or any name in
`proxide.physics.force_fields.loader.list_charmm_restricted_names()`)
resolves against it. Without that variable set, requesting CHARMM/CGenFF
content raises `proxide.assets._fetch.CharmmLicenseRequiredError` with a
pointer to this doc, rather than silently failing or (worse) fetching
restricted content on your behalf.

## `gaff/` — still vendored, unchanged

```
assets/gaff/
├── ffxml/  # OpenMM-format XML files (openmmforcefields, MIT)
└── dat/    # Original GAFF .dat parameter files
```

## Regenerating the fetch index

`_asset_index.py` is a static, committed map of bare names (e.g.
`"protein.ff14SB"`) to their relative path, so `load_force_field()` can
resolve a name without needing any files present on disk. Regenerate it
only if the upstream pinned commits in `_fetch.py` are bumped and the file
catalog changes:

```bash
uv run python scripts/sync_forcefields.py      # populate a local scratch copy
uv run python scripts/generate_asset_index.py  # rebuild _asset_index.py from it
```

## License

Fetched content is distributed under the terms of its original license
(mostly MIT, per openmmforcefields'/OpenMM's own root LICENSE — see the
spec doc for the parts of that story that are absence-of-restriction rather
than a positive grant). CHARMM/CGenFF content is never fetched or bundled;
see "Bring-your-own" above.
