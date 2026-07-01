//! Full pipeline smoke test: write a real parquet file (via arrow's own
//! `ArrowWriter`, same on-disk format a Python/pyarrow producer would emit),
//! load it through `SketchStore`, compute the distance matrix, write it out
//! as `.npy`, and read it back to confirm the round trip is lossless and
//! the values match the direct kernel computation.

use arrow::array::{Int64Array, ListArray, StringArray};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use ndarray::Array2;
use parquet::arrow::ArrowWriter;
use proxide_jaccard::{
    jaccard_distance, pairwise_jaccard_distance, write_distance_matrix, SketchStore,
};
use std::fs::File;
use std::sync::Arc;

fn write_parquet(path: &std::path::Path, rows: &[(&str, Vec<i64>)]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("accession", DataType::Utf8, false),
        Field::new(
            "hashes_list",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            true,
        ),
    ]));

    let accessions = StringArray::from(rows.iter().map(|(a, _)| *a).collect::<Vec<_>>());
    let flat: Vec<i64> = rows.iter().flat_map(|(_, h)| h.iter().copied()).collect();
    let mut offsets = vec![0i32];
    let mut acc = 0i32;
    for (_, h) in rows {
        acc += h.len() as i32;
        offsets.push(acc);
    }
    let list = ListArray::new(
        Arc::new(Field::new("item", DataType::Int64, true)),
        OffsetBuffer::new(offsets.into()),
        Arc::new(Int64Array::from(flat)),
        None,
    );

    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(accessions), Arc::new(list)]).unwrap();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[test]
fn parquet_to_npy_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let parquet_path = dir.path().join("minhashes.parquet");
    let rows: &[(&str, Vec<i64>)] = &[
        ("GCA_000002425.3", vec![5, 1, 9, 3]),
        ("GCA_000002426.1", vec![9, 3, 100, 200]),
        ("GCA_000002427.1", vec![1, 2, 3, 4, 5]),
        ("GCA_000002428.1", vec![]),
    ];
    write_parquet(&parquet_path, rows);

    let store = SketchStore::load_parquet(&parquet_path, None).unwrap();
    assert_eq!(store.len(), 4);

    let mat = pairwise_jaccard_distance(&store);

    // Cross-check against the kernel directly on the (now-sorted) inputs.
    let mut sorted_rows: Vec<Vec<i64>> = rows.iter().map(|(_, h)| h.clone()).collect();
    for r in &mut sorted_rows {
        r.sort_unstable();
    }
    for i in 0..rows.len() {
        // Diagonal is always exactly 0.0 by construction (self-distance),
        // not routed through the general kernel — which would otherwise
        // give two empty sketches a "distance" of 1.0 (0/0 convention),
        // semantically wrong for "this accession vs itself".
        assert_eq!(mat[[i, i]], 0.0, "diagonal mismatch at {i}");
        for j in (i + 1)..rows.len() {
            let expected = jaccard_distance(&sorted_rows[i], &sorted_rows[j]) as f32;
            assert_eq!(mat[[i, j]], expected, "mismatch at ({i},{j})");
            assert_eq!(mat[[j, i]], expected, "mismatch at ({j},{i})");
        }
    }

    let npy_path = dir.path().join("dist.npy");
    write_distance_matrix(&npy_path, &mat, store.accessions()).unwrap();

    let loaded: Array2<f32> = ndarray_npy::read_npy(&npy_path).unwrap();
    assert_eq!(loaded, mat);

    let manifest = std::fs::read_to_string(dir.path().join("dist.accessions.txt")).unwrap();
    assert_eq!(manifest, store.accessions().join("\n"));
}
