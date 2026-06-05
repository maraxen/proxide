use std::sync::atomic::{AtomicUsize, Ordering};

static NUM_THREADS: AtomicUsize = AtomicUsize::new(1);

pub fn set_num_threads(n: usize) {
    NUM_THREADS.store(n.max(1), Ordering::Relaxed);
}

pub fn num_threads() -> usize {
    NUM_THREADS.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_one() {
        assert_eq!(num_threads(), 1);
    }

    #[test]
    fn set_and_get() {
        set_num_threads(4);
        assert_eq!(num_threads(), 4);
        set_num_threads(1);
    }

    #[test]
    fn zero_clamps_to_one() {
        set_num_threads(0);
        assert_eq!(num_threads(), 1);
        set_num_threads(1);
    }
}
