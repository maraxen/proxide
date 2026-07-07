//! Loads the 2-column minhash parquet (`accession: Utf8`, `hashes_list: List<Int64>`
//! or `LargeList<Int64>` — the real corpus uses `large_list`) into a
//! ragged/CSR-style in-memory store: one contiguous `Vec<i64>` of all
//! hashes back to back, plus per-row offsets. This avoids one small heap
//! allocation per accession and keeps each sketch's hashes cache-contiguous
//! for the merge-intersection kernel in `distance.rs`.

use crate::error::{JaccardError, Result};
use arrow::array::{Array, AsArray, BooleanArray};
use arrow::datatypes::{DataType, Int64Type};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::{ArrowPredicateFn, ParquetRecordBatchReaderBuilder, RowFilter};
use parquet::arrow::ProjectionMask;
use parquet::data_type::AsBytes;
use parquet::file::statistics::Statistics;
use parquet::schema::types::SchemaDescriptor;
use std::collections::HashSet;
use std::fs::File;
use std::path::Path;

const ACCESSION_COL: &str = "accession";
const HASHES_COL: &str = "hashes_list";

#[derive(Debug)]
pub struct SketchStore {
    accessions: Vec<String>,
    offsets: Vec<usize>,
    hashes: Vec<i64>,
}

impl SketchStore {
    pub fn len(&self) -> usize {
        self.accessions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.accessions.is_empty()
    }

    pub fn accessions(&self) -> &[String] {
        &self.accessions
    }

    /// The sorted, deduplicated hash sketch for accession at row `idx`.
    pub fn sketch(&self, idx: usize) -> &[i64] {
        &self.hashes[self.offsets[idx]..self.offsets[idx + 1]]
    }

    /// Builds a store directly from already-fetched (accession, hashes)
    /// pairs, e.g. rows pulled one-by-one from an alternative backend (see
    /// `ipc_index`) rather than streamed from a parquet file. Sorts each
    /// row defensively, same as `load_parquet`.
    pub fn from_pairs(rows: Vec<(String, Vec<i64>)>) -> Self {
        let mut accessions = Vec::with_capacity(rows.len());
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        offsets.push(0usize);
        let mut hashes = Vec::new();
        for (accession, mut row_hashes) in rows {
            row_hashes.sort_unstable();
            hashes.extend_from_slice(&row_hashes);
            accessions.push(accession);
            offsets.push(hashes.len());
        }
        Self {
            accessions,
            offsets,
            hashes,
        }
    }

    /// Test-only constructor for exercising matrix/distance logic without a
    /// parquet round-trip. Sorts each row defensively, same as the loader.
    #[cfg(test)]
    pub fn from_rows_for_test(rows: &[(&str, &[i64])]) -> Self {
        let mut accessions = Vec::new();
        let mut offsets = vec![0usize];
        let mut hashes = Vec::new();
        for (accession, row_hashes) in rows {
            hashes.extend_from_slice(row_hashes);
            let new_len = hashes.len();
            hashes[new_len - row_hashes.len()..new_len].sort_unstable();
            accessions.push((*accession).to_string());
            offsets.push(new_len);
        }
        Self {
            accessions,
            offsets,
            hashes,
        }
    }

    /// Stream `path`, keeping only rows whose accession is in `wanted`
    /// (or every row, if `wanted` is `None`).
    ///
    /// Errors if any requested accession is never seen in the file — a
    /// silently-shrunk matrix would otherwise surprise downstream callers
    /// that index results by the requested accession order.
    pub fn load_parquet(path: &Path, wanted: Option<&[String]>) -> Result<Self> {
        let file = File::open(path).map_err(|source| JaccardError::Open {
            path: path.to_path_buf(),
            source,
        })?;
        let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

        if let Some(w) = wanted {
            // Row-group pruning: skip any row group whose accession-column
            // min/max statistics prove it can't contain any wanted
            // accession. Correct regardless of physical row order (it only
            // ever *removes* groups it can prove don't match), but its
            // payoff depends on the data: on the real corpus
            // (signature_index_k31.parquet) it's a big win — 20 row groups
            // with non-overlapping accession ranges, i.e. the file is
            // globally sorted by accession — so a clustered accession
            // subset can skip most of the file's bytes entirely.
            if let Some(keep) = prune_row_groups(&builder, w) {
                builder = builder.with_row_groups(keep);
            }

            // RowFilter (late materialization): decode only the cheap
            // `accession` column first and test membership *before* the
            // reader decodes `hashes_list` — the large, variable-length
            // column — for rows we're going to discard anyway. This helps
            // even within a row group that row-group pruning above
            // couldn't skip (e.g. one wanted row among 50,000).
            let acc_leaf = accession_leaf_index(builder.parquet_schema())?;
            let projection = ProjectionMask::leaves(builder.parquet_schema(), [acc_leaf]);
            let wanted_owned: HashSet<String> = w.iter().cloned().collect();
            let predicate = ArrowPredicateFn::new(projection, move |batch: RecordBatch| {
                let acc_arr = batch.column(0).as_string_opt::<i32>().ok_or_else(|| {
                    arrow::error::ArrowError::SchemaError(format!(
                        "{ACCESSION_COL} column is not Utf8 during row-filter evaluation"
                    ))
                })?;
                let mask: BooleanArray = (0..acc_arr.len())
                    .map(|i| !acc_arr.is_null(i) && wanted_owned.contains(acc_arr.value(i)))
                    .collect();
                Ok(mask)
            });
            builder = builder.with_row_filter(RowFilter::new(vec![Box::new(predicate)]));
        }

        let reader = builder.build()?;

        let wanted_set: Option<HashSet<&str>> =
            wanted.map(|w| w.iter().map(String::as_str).collect());
        let mut seen: HashSet<String> = HashSet::new();

        let mut accessions = Vec::new();
        let mut offsets = vec![0usize];
        let mut hashes = Vec::new();

        for batch_result in reader {
            let batch = batch_result?;

            let acc_col =
                batch
                    .column_by_name(ACCESSION_COL)
                    .ok_or(JaccardError::MissingColumn {
                        column: ACCESSION_COL,
                    })?;
            let acc_arr = acc_col
                .as_string_opt::<i32>()
                .ok_or_else(|| JaccardError::Schema {
                    column: ACCESSION_COL,
                    expected: "Utf8",
                    found: format!("{:?}", acc_col.data_type()),
                })?;

            let hashes_col = batch
                .column_by_name(HASHES_COL)
                .ok_or(JaccardError::MissingColumn { column: HASHES_COL })?;

            // The real corpus (sourmash signature_index_k31.parquet) stores
            // hashes_list as `large_list<int64>` (i64 offsets) to fit
            // ~982k accessions' worth of merged sketches; smaller/test
            // files may use plain `list<int64>` (i32 offsets). Handle both
            // — `as_list_opt::<O>()` checks the offset width matches.
            macro_rules! process_list_column {
                ($list_arr:expr) => {{
                    let list_arr = $list_arr;
                    let values = list_arr
                        .values()
                        .as_primitive_opt::<Int64Type>()
                        .ok_or_else(|| JaccardError::Schema {
                            column: HASHES_COL,
                            expected: "List<Int64> or LargeList<Int64>",
                            found: format!("{:?}", list_arr.values().data_type()),
                        })?;
                    let list_offsets = list_arr.offsets();
                    let values_slice = values.values();

                    for row in 0..batch.num_rows() {
                        if acc_arr.is_null(row) {
                            log::warn!("skipping row {row} with null accession");
                            continue;
                        }
                        let accession = acc_arr.value(row);

                        if let Some(set) = &wanted_set {
                            if !set.contains(accession) {
                                continue;
                            }
                        }

                        let row_hashes: &[i64] = if list_arr.is_null(row) {
                            &[]
                        } else {
                            let start = list_offsets[row] as usize;
                            let end = list_offsets[row + 1] as usize;
                            &values_slice[start..end]
                        };

                        hashes.extend_from_slice(row_hashes);
                        let new_len = hashes.len();
                        // Sketches are expected to already be sets (sourmash-style
                        // scaled MinHash signatures never contain duplicate hash
                        // values, and the merge pipeline sorts them); sort
                        // defensively since the kernel requires it, but don't pay
                        // for a dedup pass on a precondition the upstream format
                        // already guarantees.
                        hashes[new_len - row_hashes.len()..new_len].sort_unstable();

                        accessions.push(accession.to_string());
                        offsets.push(new_len);
                        if wanted_set.is_some() {
                            seen.insert(accession.to_string());
                        }
                    }
                }};
            }

            match hashes_col.data_type() {
                DataType::List(_) => {
                    let list_arr =
                        hashes_col
                            .as_list_opt::<i32>()
                            .ok_or_else(|| JaccardError::Schema {
                                column: HASHES_COL,
                                expected: "List<Int64>",
                                found: format!("{:?}", hashes_col.data_type()),
                            })?;
                    process_list_column!(list_arr);
                }
                DataType::LargeList(_) => {
                    let list_arr =
                        hashes_col
                            .as_list_opt::<i64>()
                            .ok_or_else(|| JaccardError::Schema {
                                column: HASHES_COL,
                                expected: "LargeList<Int64>",
                                found: format!("{:?}", hashes_col.data_type()),
                            })?;
                    process_list_column!(list_arr);
                }
                other => {
                    return Err(JaccardError::Schema {
                        column: HASHES_COL,
                        expected: "List<Int64> or LargeList<Int64>",
                        found: format!("{other:?}"),
                    });
                }
            }
        }

        if let Some(set) = &wanted_set {
            let missing: Vec<String> = set
                .iter()
                .filter(|a| !seen.contains(**a))
                .map(|a| a.to_string())
                .collect();
            if !missing.is_empty() {
                return Err(JaccardError::MissingAccessions { missing });
            }
        }

        Ok(Self {
            accessions,
            offsets,
            hashes,
        })
    }
}

fn accession_leaf_index(schema_descr: &SchemaDescriptor) -> Result<usize> {
    (0..schema_descr.num_columns())
        .find(|&i| schema_descr.column(i).name() == ACCESSION_COL)
        .ok_or(JaccardError::MissingColumn {
            column: ACCESSION_COL,
        })
}

/// Row groups whose accession-column statistics prove they can't contain
/// any accession in `wanted`. Returns `None` if pruning can't be attempted
/// at all (e.g. no statistics for the accession column anywhere) — the
/// caller then scans every row group, identical to not calling this.
/// Per-group, missing/unreadable statistics fail open (keep the group)
/// rather than risk dropping a row that might match.
fn prune_row_groups(
    builder: &ParquetRecordBatchReaderBuilder<File>,
    wanted: &[String],
) -> Option<Vec<usize>> {
    let acc_idx = accession_leaf_index(builder.parquet_schema()).ok()?;

    let mut wanted_sorted: Vec<&str> = wanted.iter().map(String::as_str).collect();
    wanted_sorted.sort_unstable();
    wanted_sorted.dedup();
    if wanted_sorted.is_empty() {
        return Some(Vec::new());
    }

    let metadata = builder.metadata();
    let mut keep = Vec::with_capacity(metadata.num_row_groups());
    for (i, rg) in metadata.row_groups().iter().enumerate() {
        let bounds = match rg.column(acc_idx).statistics() {
            Some(Statistics::ByteArray(s)) => match (s.min_opt(), s.max_opt()) {
                (Some(min), Some(max)) => std::str::from_utf8(min.as_bytes())
                    .ok()
                    .zip(std::str::from_utf8(max.as_bytes()).ok()),
                _ => None,
            },
            _ => None,
        };
        let overlaps = match bounds {
            Some((min, max)) => {
                let pos = wanted_sorted.partition_point(|w| *w < min);
                pos < wanted_sorted.len() && wanted_sorted[pos] <= max
            }
            None => true,
        };
        if overlaps {
            keep.push(i);
        }
    }
    Some(keep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, LargeListArray, ListArray, StringArray};
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::{Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    fn write_test_parquet(path: &Path, rows: &[(&str, Vec<i64>)]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new(ACCESSION_COL, DataType::Utf8, false),
            Field::new(
                HASHES_COL,
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
            RecordBatch::try_new(schema.clone(), vec![Arc::new(accessions), Arc::new(list)])
                .unwrap();

        let file = File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// Mirrors the real corpus schema (`signature_index_k31.parquet`):
    /// `hashes_list: large_list<int64>` — i64 offsets, not the plain
    /// `list<int64>` (i32 offsets) `write_test_parquet` above produces.
    fn write_test_parquet_large_list(path: &Path, rows: &[(&str, Vec<i64>)]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new(ACCESSION_COL, DataType::Utf8, false),
            Field::new(
                HASHES_COL,
                DataType::LargeList(Arc::new(Field::new("item", DataType::Int64, true))),
                true,
            ),
        ]));

        let accessions = StringArray::from(rows.iter().map(|(a, _)| *a).collect::<Vec<_>>());
        let flat: Vec<i64> = rows.iter().flat_map(|(_, h)| h.iter().copied()).collect();
        let mut offsets = vec![0i64];
        let mut acc = 0i64;
        for (_, h) in rows {
            acc += h.len() as i64;
            offsets.push(acc);
        }
        let list = LargeListArray::new(
            Arc::new(Field::new("item", DataType::Int64, true)),
            OffsetBuffer::new(offsets.into()),
            Arc::new(Int64Array::from(flat)),
            None,
        );

        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(accessions), Arc::new(list)])
                .unwrap();

        let file = File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// Writes one row group per entry in `groups` (via explicit `flush()`
    /// between batches), so tests can exercise `prune_row_groups` and the
    /// `RowFilter` path against a file shaped like the real corpus
    /// (multiple row groups with distinct accession ranges) instead of the
    /// single-row-group fixtures `write_test_parquet*` produce.
    fn write_test_parquet_multi_row_group(path: &Path, groups: &[&[(&str, Vec<i64>)]]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new(ACCESSION_COL, DataType::Utf8, false),
            Field::new(
                HASHES_COL,
                DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                true,
            ),
        ]));

        let file = File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();
        for rows in groups {
            let accessions = StringArray::from(rows.iter().map(|(a, _)| *a).collect::<Vec<_>>());
            let flat: Vec<i64> = rows.iter().flat_map(|(_, h)| h.iter().copied()).collect();
            let mut offsets = vec![0i32];
            let mut acc = 0i32;
            for (_, h) in *rows {
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
                RecordBatch::try_new(schema.clone(), vec![Arc::new(accessions), Arc::new(list)])
                    .unwrap();
            writer.write(&batch).unwrap();
            writer.flush().unwrap();
        }
        writer.close().unwrap();
    }

    #[test]
    fn multi_row_group_query_spanning_one_group_is_correct() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi_rg.parquet");
        // Three row groups with non-overlapping, sorted accession ranges —
        // shaped like the real signature_index_k31.parquet layout.
        write_test_parquet_multi_row_group(
            &path,
            &[
                &[
                    ("GCA_000000001.1", vec![1, 2]),
                    ("GCA_000000002.1", vec![3, 4]),
                ],
                &[
                    ("GCA_000000010.1", vec![5, 6]),
                    ("GCA_000000011.1", vec![7, 8]),
                ],
                &[
                    ("GCF_000000020.1", vec![9, 10]),
                    ("GCF_000000021.1", vec![11, 12]),
                ],
            ],
        );

        // Sanity: the fixture really did produce 3 row groups, otherwise
        // this test wouldn't actually exercise row-group pruning.
        let file = File::open(&path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        assert_eq!(builder.metadata().num_row_groups(), 3);

        // Wanted accessions fall entirely within row group 1 (middle).
        let wanted = vec!["GCA_000000010.1".to_string(), "GCA_000000011.1".to_string()];
        let store = SketchStore::load_parquet(&path, Some(&wanted)).unwrap();
        assert_eq!(store.len(), 2);
        let mut found: Vec<&str> = store.accessions().iter().map(String::as_str).collect();
        found.sort_unstable();
        assert_eq!(found, vec!["GCA_000000010.1", "GCA_000000011.1"]);

        // Wanted accessions span the first and last row groups, skipping
        // the middle one entirely — pruning must not drop them.
        let wanted_spanning = vec!["GCA_000000001.1".to_string(), "GCF_000000021.1".to_string()];
        let store2 = SketchStore::load_parquet(&path, Some(&wanted_spanning)).unwrap();
        assert_eq!(store2.len(), 2);
        let mut found2: Vec<&str> = store2.accessions().iter().map(String::as_str).collect();
        found2.sort_unstable();
        assert_eq!(found2, vec!["GCA_000000001.1", "GCF_000000021.1"]);
    }

    #[test]
    fn loads_large_list_schema_like_the_real_corpus() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_large_list.parquet");
        write_test_parquet_large_list(
            &path,
            &[
                ("GCA_000002425.3", vec![5, 3, 1]),
                ("GCA_000002426.1", vec![]),
                ("GCA_000002427.1", vec![10, 20, 30]),
            ],
        );

        let store = SketchStore::load_parquet(&path, None).unwrap();
        assert_eq!(store.len(), 3);
        assert_eq!(store.sketch(0), &[1, 3, 5]);
        assert_eq!(store.sketch(1), &[] as &[i64]);
        assert_eq!(store.sketch(2), &[10, 20, 30]);
    }

    #[test]
    fn loads_all_rows_when_unfiltered() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        write_test_parquet(
            &path,
            &[
                ("GCA_000002425.3", vec![5, 3, 1]),
                ("GCA_000002426.1", vec![]),
                ("GCA_000002427.1", vec![10, 20, 30]),
            ],
        );

        let store = SketchStore::load_parquet(&path, None).unwrap();
        assert_eq!(store.len(), 3);
        assert_eq!(store.sketch(0), &[1, 3, 5]); // sorted defensively
        assert_eq!(store.sketch(1), &[] as &[i64]);
        assert_eq!(store.sketch(2), &[10, 20, 30]);
        assert_eq!(store.accessions()[0], "GCA_000002425.3");
    }

    #[test]
    fn filters_to_requested_accessions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        write_test_parquet(
            &path,
            &[
                ("GCA_000002425.3", vec![1, 2, 3]),
                ("GCA_000002426.1", vec![4, 5, 6]),
                ("GCA_000002427.1", vec![7, 8, 9]),
            ],
        );

        let wanted = vec!["GCA_000002425.3".to_string(), "GCA_000002427.1".to_string()];
        let store = SketchStore::load_parquet(&path, Some(&wanted)).unwrap();
        assert_eq!(store.len(), 2);
        assert_eq!(store.accessions(), &wanted[..]);
    }

    #[test]
    fn errors_on_missing_requested_accession() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        write_test_parquet(&path, &[("GCA_000002425.3", vec![1, 2, 3])]);

        let wanted = vec!["GCA_000002425.3".to_string(), "GCA_999999999.1".to_string()];
        let err = SketchStore::load_parquet(&path, Some(&wanted)).unwrap_err();
        match err {
            JaccardError::MissingAccessions { missing } => {
                assert_eq!(missing, vec!["GCA_999999999.1".to_string()]);
            }
            other => panic!("expected MissingAccessions, got {other:?}"),
        }
    }
}
