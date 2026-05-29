/// Test flanking behavior in backbone interaction computation.
///
/// This test is marked as ignored because proper testing requires:
/// 1. A real rotamer library (ConFind::new requires Arc<RotamerLibrary>)
/// 2. A populated backbone with actual rotamer caching
///
/// To run this test, you would need to:
/// 1. Load a rotamer library from disk
/// 2. Create a synthetic or real protein backbone with known geometry
/// 3. Construct a ConFind instance
/// 4. Test that bb_interaction respects ignore_flanking parameter
///
/// The expected behavior is:
/// - With ignore_flanking=1: adjacent residues (i, i+1) on same chain not reported
/// - With ignore_flanking=1: residues (i, i+2) on same chain still reported
/// - With ignore_flanking=0: all residues within distance are reported
/// - With different chains: flanking distance is ignored (all within dcut reported)

#[test]
#[ignore = "requires rotamer library and backbone construction"]
fn flanking_same_chain() {
    // Placeholder for flanking test.
    // Implementation would require:
    // - ConFind instance with real rotlib
    // - Test backbone with known CA positions
    // - bb_interaction() calls with different ignore_flanking values
}
