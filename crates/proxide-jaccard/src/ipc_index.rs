//! **Prototype** — not wired into the CLI. Evaluates an Arrow IPC file
//! (uncompressed) + a sorted in-memory accession index as an alternative
//! to the parquet path in `sketch.rs`.
//!
//! Why: parquet's row-group/page granularity means a scattered point-query
//! across the corpus still pays to decode every touched group (measured:
//! 63.82s / 205MB RSS for 300 accessions scattered across all 20 row
//! groups of the real corpus, vs 0.16s / 57MB when the same count of
//! accessions clusters in one group). Arrow's on-disk IPC format *is* its
//! in-memory layout — no decode step — so once we know which (batch, row)
//! an accession lives at, fetching it costs exactly one seek + one
//! (typically small) record-batch read, regardless of how scattered the
//! overall query is. The tradeoff is disk: this format has no built-in
//! compression, and the underlying hash data measured ~73% incompressible
//! under zstd (3.67B near-uniform-random int64 values across the real
//! corpus), so storing it raw costs roughly +37% disk vs the current
//! parquet file.
//!
//! See `.praxia/docs/research/260630_arrow-ipc-prototype.md` for the
//! measured numbers this module produced.

use crate::error::{JaccardError, Result};
use arrow::array::{Array, AsArray};
use arrow::datatypes::{DataType, Int64Type, SchemaRef};
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const ACCESSION_COL: &str = "accession";
const HASHES_COL: &str = "hashes_list";

/// Sorted `accession -> (ipc batch index, row within that batch)`.
/// Small enough to hold fully in memory and serialize as a tiny sidecar
/// next to the IPC file (a few tens of MB for the full ~982k-row corpus,
/// vs. 20-27GB for the hash data itself).
#[derive(Clone, Serialize, Deserialize)]
pub struct AccessionIndex {
    /// Sorted by accession (binary_search requires this).
    entries: Vec<(String, u32, u32)>,
}

impl AccessionIndex {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn lookup(&self, accession: &str) -> Option<(u32, u32)> {
        self.entries
            .binary_search_by(|(a, _, _)| a.as_str().cmp(accession))
            .ok()
            .map(|i| (self.entries[i].1, self.entries[i].2))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let file = File::create(path).map_err(|source| JaccardError::Open {
            path: path.to_path_buf(),
            source,
        })?;
        bincode::serialize_into(BufWriter::new(file), self).map_err(|e| JaccardError::Serialize {
            path: path.to_path_buf(),
            message: e.to_string(),
        })
    }

    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|source| JaccardError::Open {
            path: path.to_path_buf(),
            source,
        })?;
        bincode::deserialize_from(file).map_err(|e| JaccardError::Serialize {
            path: path.to_path_buf(),
            message: e.to_string(),
        })
    }
}

/// Streams `parquet_path` into a single-schema, uncompressed Arrow IPC
/// file at `ipc_path` — `batch_size` rows per IPC record batch, which is
/// also the unit of random access (smaller batches mean finer-grained
/// lookups but more index entries / footer overhead). Optionally
/// restricts to specific parquet row groups (for prototyping against a
/// bounded slice of a large corpus without materializing all of it).
///
/// Streaming: never holds more than one batch's worth of rows in memory
/// at a time, regardless of corpus size — unlike loading everything into
/// a `SketchStore`, this is safe to run against the full corpus on a
/// memory-constrained machine.
pub fn convert_parquet_to_ipc(
    parquet_path: &Path,
    ipc_path: &Path,
    batch_size: usize,
    row_groups: Option<Vec<usize>>,
) -> Result<AccessionIndex> {
    let file = File::open(parquet_path).map_err(|source| JaccardError::Open {
        path: parquet_path.to_path_buf(),
        source,
    })?;
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)?.with_batch_size(batch_size);
    if let Some(rgs) = row_groups {
        builder = builder.with_row_groups(rgs);
    }
    // Passthrough schema (whatever the source actually is — List or
    // LargeList hashes_list — rather than assuming one, see sketch.rs).
    let schema: SchemaRef = builder.schema().clone();
    let reader = builder.build()?;

    let out = File::create(ipc_path).map_err(|source| JaccardError::Open {
        path: ipc_path.to_path_buf(),
        source,
    })?;
    let mut writer = FileWriter::try_new(BufWriter::new(out), schema.as_ref())?;

    let mut entries = Vec::new();
    for (batch_idx, batch_result) in reader.enumerate() {
        let batch = batch_result?;
        let acc_col = batch
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
        for row in 0..batch.num_rows() {
            if !acc_arr.is_null(row) {
                entries.push((acc_arr.value(row).to_string(), batch_idx as u32, row as u32));
            }
        }
        writer.write(&batch)?;
    }
    writer.finish()?;

    entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    Ok(AccessionIndex { entries })
}

/// Point-lookup reader over an IPC file built by [`convert_parquet_to_ipc`].
///
/// Caches the most recently decoded batch: when consecutive lookups land
/// in the same batch (the common case for a clustered accession set,
/// where many wanted rows share one batch), only the first one pays the
/// decode cost. Without this, a clustered query of N accessions sharing
/// a batch would redundantly decode that batch N times — measured at
/// ~9s for N=300 on the real corpus before this cache was added, vs.
/// parquet+RowFilter's 0.16s for an equivalent clustered query.
pub struct IpcSketchReader {
    reader: FileReader<File>,
    index: AccessionIndex,
    cached_batch: Option<(usize, arrow::record_batch::RecordBatch)>,
}

impl IpcSketchReader {
    pub fn open(ipc_path: &Path, index: AccessionIndex) -> Result<Self> {
        let file = File::open(ipc_path).map_err(|source| JaccardError::Open {
            path: ipc_path.to_path_buf(),
            source,
        })?;
        let reader = FileReader::try_new(file, None)?;
        Ok(Self {
            reader,
            index,
            cached_batch: None,
        })
    }

    /// Fetches one accession's sorted hash sketch. Pays a seek + batch
    /// decode only when the requested row isn't in the already-cached
    /// batch from the previous call.
    pub fn get(&mut self, accession: &str) -> Result<Option<Vec<i64>>> {
        let Some((batch_idx, row)) = self.index.lookup(accession) else {
            return Ok(None);
        };
        let batch_idx = batch_idx as usize;

        if self.cached_batch.as_ref().map(|(idx, _)| *idx) != Some(batch_idx) {
            self.reader.set_index(batch_idx)?;
            let batch = self.reader.next().ok_or_else(|| JaccardError::Schema {
                column: HASHES_COL,
                expected: "a record batch at the indexed position",
                found: "none".to_string(),
            })??;
            self.cached_batch = Some((batch_idx, batch));
        }
        let batch = &self.cached_batch.as_ref().unwrap().1;

        let hashes_col = batch
            .column_by_name(HASHES_COL)
            .ok_or(JaccardError::MissingColumn { column: HASHES_COL })?;
        let row = row as usize;
        let hashes = match hashes_col.data_type() {
            DataType::List(_) => {
                let list_arr =
                    hashes_col
                        .as_list_opt::<i32>()
                        .ok_or_else(|| JaccardError::Schema {
                            column: HASHES_COL,
                            expected: "List<Int64>",
                            found: format!("{:?}", hashes_col.data_type()),
                        })?;
                extract_row::<i32>(list_arr, row)?
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
                extract_row::<i64>(list_arr, row)?
            }
            other => {
                return Err(JaccardError::Schema {
                    column: HASHES_COL,
                    expected: "List<Int64> or LargeList<Int64>",
                    found: format!("{other:?}"),
                });
            }
        };
        Ok(Some(hashes))
    }

    /// Fetches every accession in `wanted`, building a [`crate::SketchStore`]
    /// ready for [`crate::pairwise_jaccard_distance`]. Errors (same
    /// contract as `SketchStore::load_parquet`) if any requested
    /// accession isn't in the index.
    pub fn get_many(&mut self, wanted: &[String]) -> Result<crate::SketchStore> {
        let mut rows = Vec::with_capacity(wanted.len());
        let mut missing = Vec::new();
        for accession in wanted {
            match self.get(accession)? {
                Some(hashes) => rows.push((accession.clone(), hashes)),
                None => missing.push(accession.clone()),
            }
        }
        if !missing.is_empty() {
            return Err(JaccardError::MissingAccessions { missing });
        }
        Ok(crate::SketchStore::from_pairs(rows))
    }
}

fn extract_row<O>(list_arr: &arrow::array::GenericListArray<O>, row: usize) -> Result<Vec<i64>>
where
    O: arrow::array::OffsetSizeTrait,
{
    if list_arr.is_null(row) {
        return Ok(Vec::new());
    }
    let values = list_arr
        .values()
        .as_primitive_opt::<Int64Type>()
        .ok_or_else(|| JaccardError::Schema {
            column: HASHES_COL,
            expected: "List<Int64> or LargeList<Int64>",
            found: format!("{:?}", list_arr.values().data_type()),
        })?;
    let offsets = list_arr.offsets();
    let start = offsets[row].as_usize();
    let end = offsets[row + 1].as_usize();
    let mut hashes = values.values()[start..end].to_vec();
    hashes.sort_unstable();
    Ok(hashes)
}

/// Convenience used by the benchmark example: resolve which parquet row
/// groups a set of accessions could fall in, via the same min/max
/// statistics `sketch.rs::prune_row_groups` uses — lets the prototype
/// convert only the row groups actually needed for a given accession set
/// instead of always touching the whole corpus.
pub fn row_groups_for_accessions(parquet_path: &Path, wanted: &[String]) -> Result<Vec<usize>> {
    let file = File::open(parquet_path).map_err(|source| JaccardError::Open {
        path: parquet_path.to_path_buf(),
        source,
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema_descr = builder.parquet_schema();
    let acc_idx = (0..schema_descr.num_columns())
        .find(|&i| schema_descr.column(i).name() == ACCESSION_COL)
        .ok_or(JaccardError::MissingColumn {
            column: ACCESSION_COL,
        })?;

    let wanted_set: HashSet<&str> = wanted.iter().map(String::as_str).collect();
    let metadata = builder.metadata();
    let mut keep = Vec::new();
    for (i, rg) in metadata.row_groups().iter().enumerate() {
        use parquet::data_type::AsBytes;
        use parquet::file::statistics::Statistics;
        let overlaps = match rg.column(acc_idx).statistics() {
            Some(Statistics::ByteArray(s)) => match (s.min_opt(), s.max_opt()) {
                (Some(min), Some(max)) => {
                    let (min, max) = (
                        std::str::from_utf8(min.as_bytes()).unwrap_or(""),
                        std::str::from_utf8(max.as_bytes()).unwrap_or(""),
                    );
                    wanted_set.iter().any(|w| *w >= min && *w <= max)
                }
                _ => true,
            },
            _ => true,
        };
        if overlaps {
            keep.push(i);
        }
    }
    Ok(keep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sketch::SketchStore;
    use arrow::array::{Int64Array, ListArray, StringArray};
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::{Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    fn write_test_parquet(path: &Path, groups: &[&[(&str, Vec<i64>)]]) {
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
    fn ipc_round_trip_matches_parquet_path() {
        let dir = tempfile::tempdir().unwrap();
        let parquet_path = dir.path().join("src.parquet");
        let ipc_path = dir.path().join("out.arrow");

        write_test_parquet(
            &parquet_path,
            &[
                &[
                    ("GCA_000000001.1", vec![5, 1, 3]),
                    ("GCA_000000002.1", vec![]),
                ],
                &[
                    ("GCF_000000010.1", vec![9, 7]),
                    ("GCF_000000011.1", vec![2, 2, 4]),
                ],
            ],
        );

        let index = convert_parquet_to_ipc(&parquet_path, &ipc_path, 1, None).unwrap();
        assert_eq!(index.len(), 4);

        let mut reader = IpcSketchReader::open(&ipc_path, index).unwrap();
        assert_eq!(reader.get("GCA_000000001.1").unwrap(), Some(vec![1, 3, 5]));
        assert_eq!(reader.get("GCA_000000002.1").unwrap(), Some(vec![]));
        assert_eq!(reader.get("nonexistent").unwrap(), None);

        // Cross-check against the parquet path for the same accessions.
        let wanted = vec!["GCA_000000001.1".to_string(), "GCF_000000011.1".to_string()];
        let via_ipc = reader.get_many(&wanted).unwrap();
        let via_parquet = SketchStore::load_parquet(&parquet_path, Some(&wanted)).unwrap();
        for i in 0..wanted.len() {
            assert_eq!(via_ipc.sketch(i), via_parquet.sketch(i));
        }
    }

    #[test]
    fn index_serializes_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let index_path = dir.path().join("index.bin");
        let index = AccessionIndex {
            entries: vec![("a".to_string(), 0, 0), ("b".to_string(), 0, 1)],
        };
        index.save(&index_path).unwrap();
        let loaded = AccessionIndex::load(&index_path).unwrap();
        assert_eq!(loaded.lookup("a"), Some((0, 0)));
        assert_eq!(loaded.lookup("b"), Some((0, 1)));
        assert_eq!(loaded.lookup("c"), None);
    }

    #[test]
    fn missing_accession_errors_like_parquet_path() {
        let dir = tempfile::tempdir().unwrap();
        let parquet_path = dir.path().join("src.parquet");
        let ipc_path = dir.path().join("out.arrow");
        write_test_parquet(&parquet_path, &[&[("GCA_000000001.1", vec![1])]]);

        let index = convert_parquet_to_ipc(&parquet_path, &ipc_path, 10, None).unwrap();
        let mut reader = IpcSketchReader::open(&ipc_path, index).unwrap();
        let wanted = vec!["GCA_000000001.1".to_string(), "GCA_999999999.1".to_string()];
        let err = reader.get_many(&wanted).unwrap_err();
        match err {
            JaccardError::MissingAccessions { missing } => {
                assert_eq!(missing, vec!["GCA_999999999.1".to_string()]);
            }
            other => panic!("expected MissingAccessions, got {other:?}"),
        }
    }
}
