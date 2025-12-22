use super::read_foldcomp_from_reader;
use crate::structure::systems::AtomicSystem;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

// MMseqs2 database structure:
// .db: Concatenated data entries
// .index: TSV file with lines: ID <tab> Offset <tab> Length
// .lookup: TSV file with lines: Key(ID) <tab> Name

#[allow(dead_code)]
pub struct FoldCompDb {
    pub keys: Vec<u32>,               // IDs in order
    pub offsets: Vec<u64>,            // Offsets in .db file
    pub lengths: Vec<u64>,            // Lengths in .db file
    pub lookup: HashMap<String, u32>, // Name -> ID
    db_path: PathBuf,
}

impl FoldCompDb {
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let db_path = path.as_ref().to_path_buf();
        let base = db_path.to_string_lossy().to_string();

        let index_path = format!("{}.index", base);
        let lookup_path = format!("{}.lookup", base);

        let (keys, offsets, lengths) = Self::read_index(&index_path)?;
        let lookup = Self::read_lookup(&lookup_path)?;

        Ok(Self {
            keys,
            offsets,
            lengths,
            lookup,
            db_path,
        })
    }

    fn read_index(path: &str) -> std::io::Result<(Vec<u32>, Vec<u64>, Vec<u64>)> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut keys = Vec::new();
        let mut offsets = Vec::new();
        let mut lengths = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() < 3 {
                continue;
            }

            let id = parts[0]
                .parse::<u32>()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let offset = parts[1]
                .parse::<u64>()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let length = parts[2]
                .parse::<u64>()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            keys.push(id);
            offsets.push(offset);
            lengths.push(length);
        }

        Ok((keys, offsets, lengths))
    }

    fn read_lookup(path: &str) -> std::io::Result<HashMap<String, u32>> {
        let mut map = HashMap::new();
        if !Path::new(path).exists() {
            return Ok(map); // Optional?
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            // Format: ID <tab> Name
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() < 2 {
                continue;
            }

            let id = parts[0]
                .parse::<u32>()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let name = parts[1].to_string();

            map.insert(name, id);
        }

        Ok(map)
    }

    pub fn get(&self, id: u32) -> std::io::Result<AtomicSystem> {
        // Find index of ID in self.keys
        // Assuming sorted? Or just linear scan/binary search?
        // Actually .index is usually sorted by ID, but let's binary search.

        let idx = match self.keys.binary_search(&id) {
            Ok(i) => i,
            Err(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "ID not found",
                ))
            }
        };

        let offset = self.offsets[idx];
        let length = self.lengths[idx];

        // Open file and read
        let mut file = File::open(&self.db_path)?;
        file.seek(SeekFrom::Start(offset))?;

        // We need a reader limited to `length` bytes.
        // But `read_foldcomp_from_reader` takes `&mut R`.
        // A generic `Take` doesn't implement Seek.
        // However, `read_foldcomp_from_reader` does forward seeks (SeekFrom::Current) which works if we track position.
        // But checking `read_foldcomp_from_reader`, it uses `seek` to skip sections.

        // If we just pass `file`, it might seek past the end of the chunk if the file is corrupt or validly?
        // If we seek relative to current, it works on the file.
        // We just need to ensure we start at `offset`.
        // The reader within `read_foldcomp_from_reader` will seek.
        // As long as it doesn't need to seek relative to Start or End of logical file (which it doesn't seem to do, mostly skips).

        // Wait, `read_foldcomp_from_reader` calls `reader.seek(SeekFrom::Current(...))`.
        // It does NOT call SeekFrom::Start or End.
        // So passing the File handle (already seeked to offset) is safe-ish,
        // assuming `read_foldcomp_from_reader` consumes exactly what it needs.
        // But what if it over-reads?
        // We can create a bounded reader wrapper that supports seek?

        // For now, let's just pass `file`. The compressed format is self-describing (header has sizes),
        // so it should stop correctly. But we should be careful.
        // Actually, we can read the whole chunk into a Cursor<Vec<u8>> since `length` is small (6KB).
        // This is safer and fast.

        let mut buffer = vec![0u8; length as usize];
        file.read_exact(&mut buffer)?;

        let mut cursor = std::io::Cursor::new(buffer);
        read_foldcomp_from_reader(&mut cursor)
    }

    pub fn get_by_name(&self, name: &str) -> std::io::Result<AtomicSystem> {
        if let Some(&id) = self.lookup.get(name) {
            self.get(id)
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Name {} not found", name),
            ))
        }
    }
}
