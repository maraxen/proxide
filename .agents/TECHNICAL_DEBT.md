# Proxide Technical Debt Tracker

## ✅ Recently Completed (Dec 2025)

### Phase 5: Python/Biotite Migration ✅ COMPLETE

- [x] PQR parsing, mass assignment, and structure processing ported to Rust.
- [x] Biotite dependency minimized.

### OpenMM Export & Physics ✅ COMPLETE

- [x] `to_openmm_system()` supports all standard MD forces.
- [x] CMAP support implemented.
- [x] 1-3/1-4 exclusions implemented.

## 🔴 High Priority

### Trajectory Format Integration

- **Status**: XTC working, DCD/TRR need pure-Rust alternatives.
- **Goal**: Replace `chemfiles` dependent code to resolve SIGFPE crashes.

## 🟡 Medium Priority

### Documentation Refresh

- [ ] Update `docs/` folder with current API.
- [ ] Add examples for `add_hydrogens` and MD parameterization.

### Test Suite Cleanup

- [ ] Update outdated tests in `tests/io/parsing/`.
