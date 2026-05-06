
#[cfg(test)]
mod tests {
    use crate::formats::dcd::DcdReader;
    use std::io::Cursor;

    #[test]
    fn test_dcd_invalid_magic() {
        let data = b"NOTACORD";
        let file = Cursor::new(data);
        let result = DcdReader::new(file);
        assert!(result.is_err());
    }

    #[test]
    fn test_dcd_header_parsing() {
        // Build a mock DCD header
        let mut data = Vec::new();
        // 84 (record length)
        data.extend_from_slice(&84i32.to_le_bytes());
        // CORD
        data.extend_from_slice(b"CORD");
        // Header (80 bytes)
        let mut header = [0u8; 80];
        // n_frames = 1
        header[0..4].copy_from_slice(&1i32.to_le_bytes());
        // start_step = 0
        header[4..8].copy_from_slice(&0i32.to_le_bytes());
        // save_freq = 1
        header[8..12].copy_from_slice(&1i32.to_le_bytes());
        // delta = 0.001
        header[36..40].copy_from_slice(&0.001f32.to_le_bytes());
        // has_unit_cell = 0
        header[40..44].copy_from_slice(&0i32.to_le_bytes());
        // charmm_version = 24
        header[76..80].copy_from_slice(&24i32.to_le_bytes());
        data.extend_from_slice(&header);
        // 84 (record length)
        data.extend_from_slice(&84i32.to_le_bytes());

        // 4 (title block length)
        data.extend_from_slice(&4i32.to_le_bytes());
        // Title block
        data.extend_from_slice(b"Test");
        // 4 (title block length)
        data.extend_from_slice(&4i32.to_le_bytes());

        // 4 (n_atoms record length)
        data.extend_from_slice(&4i32.to_le_bytes());
        // n_atoms = 10
        data.extend_from_slice(&10i32.to_le_bytes());
        // 4 (n_atoms record length)
        data.extend_from_slice(&4i32.to_le_bytes());

        let file = Cursor::new(data);
        let reader = DcdReader::new(file).expect("Failed to create DcdReader");
        assert_eq!(reader.header.n_frames, 1);
        assert_eq!(reader.header.n_atoms, 10);
        assert_eq!(reader.header.has_unit_cell, false);
    }
}
