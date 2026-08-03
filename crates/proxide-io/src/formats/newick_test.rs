#[cfg(test)]
mod tests {
    use crate::formats::newick::parse_newick;
    use std::fs::{remove_file, File};
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempFile(PathBuf);

    impl TempFile {
        fn new(content: &str) -> Self {
            let id = COUNTER.fetch_add(1, Ordering::SeqCst);
            let path = PathBuf::from(format!("/tmp/proxide_newick_{}.nwk", id));
            let mut file = File::create(&path).unwrap();
            write!(file, "{}", content).unwrap();
            TempFile(path)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = remove_file(&self.0);
        }
    }

    #[test]
    fn test_parse_basic() {
        let tmp = TempFile::new("(A,B);");
        let tree = parse_newick(tmp.path()).unwrap();
        assert_eq!(tree.n_leaves, 2);
        assert_eq!(tree.leaf_names.len(), 2);
        assert!(tree.leaf_names.contains(&"A".to_string()));
        assert!(tree.leaf_names.contains(&"B".to_string()));
    }

    #[test]
    fn test_parse_branch_lengths() {
        let tmp = TempFile::new("(A:0.1,B:0.2);");
        let tree = parse_newick(tmp.path()).unwrap();
        assert_eq!(tree.n_leaves, 2);
    }

    #[test]
    fn test_parse_nested() {
        let tmp = TempFile::new("((A,B),C);");
        let tree = parse_newick(tmp.path()).unwrap();
        assert_eq!(tree.n_leaves, 3);
    }

    #[test]
    fn test_parse_labels() {
        let tmp = TempFile::new("(A,B)Root;");
        let tree = parse_newick(tmp.path()).unwrap();
        assert_eq!(tree.n_leaves, 2);
    }

    #[test]
    fn test_invalid_format() {
        let tmp = TempFile::new("(A,B)");
        let result = parse_newick(tmp.path());
        assert!(result.is_err());
    }
}
