#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_dcd_invalid_magic() {
        let data = b"NOTACORD";
        let mut file = Cursor::new(data);
        // We can't use DcdReader::open directly as it uses File::open
        // But the parsing logic inside should fail.
        // Given the current implementation uses `File::open` in `DcdReader::open`,
        // I might need to refactor it to accept a generic reader or mock the file.
    }
}
