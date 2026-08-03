use proxide_frag::*;
use std::env;

#[test]
fn test_persist_roundtrip() {
    // Create a database with N=5
    let mut builder = FragmentDbBuilder::<5>::new();

    // Create a simple fragment
    let coords = [
        [
            [0.0, 0.0, 10.0],
            [1.0, 0.0, 10.0],
            [1.0, 1.0, 10.0],
            [1.0, 1.0, 11.0],
        ],
        [
            [2.0, 0.0, 10.0],
            [3.0, 0.0, 10.0],
            [3.0, 1.0, 10.0],
            [3.0, 1.0, 11.0],
        ],
        [
            [4.0, 0.0, 10.0],
            [5.0, 0.0, 10.0],
            [5.0, 1.0, 10.0],
            [5.0, 1.0, 11.0],
        ],
        [
            [6.0, 0.0, 10.0],
            [7.0, 0.0, 10.0],
            [7.0, 1.0, 10.0],
            [7.0, 1.0, 11.0],
        ],
        [
            [8.0, 0.0, 10.0],
            [9.0, 0.0, 10.0],
            [9.0, 1.0, 10.0],
            [9.0, 1.0, 11.0],
        ],
    ];

    let frag = Fragment::new(coords);
    let label = SourceLabel::new("1abc", 'A', 1, ' ', 5, ' ');
    builder
        .add_fragment(frag, label.clone())
        .expect("Failed to add fragment");

    let db = builder.build();
    assert_eq!(db.len(), 1, "Database should have 1 entry");

    // Save to temp file
    let temp_path = env::temp_dir().join("proxide_frag_persist_roundtrip_integration.bin");
    db.save(&temp_path).expect("Failed to save");

    // Load it back
    let loaded: FragmentDb<5> = FragmentDb::load(&temp_path).expect("Failed to load");
    assert_eq!(loaded.len(), 1, "Loaded database should have 1 entry");

    // Clean up
    let _ = std::fs::remove_file(&temp_path);
}

#[test]
fn test_persist_arity_mismatch() {
    // Create a database with N=5
    let mut builder = FragmentDbBuilder::<5>::new();

    let coords = [
        [
            [0.0, 0.0, 10.0],
            [1.0, 0.0, 10.0],
            [1.0, 1.0, 10.0],
            [1.0, 1.0, 11.0],
        ],
        [
            [2.0, 0.0, 10.0],
            [3.0, 0.0, 10.0],
            [3.0, 1.0, 10.0],
            [3.0, 1.0, 11.0],
        ],
        [
            [4.0, 0.0, 10.0],
            [5.0, 0.0, 10.0],
            [5.0, 1.0, 10.0],
            [5.0, 1.0, 11.0],
        ],
        [
            [6.0, 0.0, 10.0],
            [7.0, 0.0, 10.0],
            [7.0, 1.0, 10.0],
            [7.0, 1.0, 11.0],
        ],
        [
            [8.0, 0.0, 10.0],
            [9.0, 0.0, 10.0],
            [9.0, 1.0, 10.0],
            [9.0, 1.0, 11.0],
        ],
    ];

    let frag = Fragment::new(coords);
    let label = SourceLabel::new("1abc", 'A', 1, ' ', 5, ' ');
    builder
        .add_fragment(frag, label)
        .expect("Failed to add fragment");

    let db = builder.build();

    // Save to temp file
    let temp_path = env::temp_dir().join("proxide_frag_persist_arity_mismatch_integration.bin");
    db.save(&temp_path).expect("Failed to save");

    // Try to load with N=4 (wrong arity)
    let result: Result<FragmentDb<4>, _> = FragmentDb::load(&temp_path);
    assert!(result.is_err(), "Loading with wrong arity should fail");

    match result {
        Err(PersistError::ArityMismatch {
            expected: 4,
            found: 5,
        }) => {
            // Correct error
        }
        _ => panic!("Expected ArityMismatch error"),
    }

    // Clean up
    let _ = std::fs::remove_file(&temp_path);
}
