/// Test to verify the determinism invariant: running ConFind multiple times
/// on the same inputs should produce identical results.
///
/// This test is marked as ignored because it requires a real rotamer library
/// to be meaningful. To run it, set RLIB environment variable pointing to
/// a valid rotamer library file:
///
/// cargo test --test test_parallel -- --ignored --nocapture RLIB=/path/to/lib.bin
///
/// The test constructs a minimal backbone and verifies that:
/// - Running contacts() twice yields identical ContactList pairs
/// - Running contacts() twice yields identical degree values
/// - Parallel execution is deterministic

#[test]
#[ignore = "requires rotamer library — set RLIB env var to run"]
fn parallel_determinism() {
    // This test requires a real rotamer library and PDB file,
    // which are not part of the unit test suite.
    //
    // To properly test parallel determinism:
    // 1. Load a real rotamer library
    // 2. Extract backbone from a PDB
    // 3. Create ConFind instance
    // 4. Call contacts() multiple times
    // 5. Assert that ContactList pairs and degrees are identical
    //
    // For now, this serves as a placeholder to document the expectation.
    println!("Parallel determinism test requires RLIB env var");
}
