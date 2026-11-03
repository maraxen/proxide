use std::sync::atomic::{AtomicUsize, Ordering};

/// `0` means the caller has not chosen a count (`None` from
/// [`num_threads_override`]). [`num_threads`] still reports 1 in that state
/// so existing call sites stay sequential-by-default.
static NUM_THREADS: AtomicUsize = AtomicUsize::new(0);

pub fn set_num_threads(n: usize) {
    NUM_THREADS.store(n.max(1), Ordering::Relaxed);
}

/// Restore the process default: no explicit thread count.
pub fn clear_num_threads() {
    NUM_THREADS.store(0, Ordering::Relaxed);
}

pub fn num_threads() -> usize {
    NUM_THREADS.load(Ordering::Relaxed).max(1)
}

/// `Some(n)` after [`set_num_threads`] (`n >= 1`; `0` is clamped to 1).
/// `None` when the count is still the process default.
pub fn num_threads_override() -> Option<usize> {
    match NUM_THREADS.load(Ordering::Relaxed) {
        0 => None,
        n => Some(n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    // NUM_THREADS is process-global and cargo runs these tests on parallel threads in one
    // binary, so each test holds this lock for its whole body (debt #2915).
    static GLOBAL: Mutex<()> = Mutex::new(());

    fn lock() -> MutexGuard<'static, ()> {
        GLOBAL.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn default_is_one() {
        let _guard = lock();
        assert_eq!(num_threads(), 1);
    }

    #[test]
    fn set_and_get() {
        let _guard = lock();
        set_num_threads(4);
        assert_eq!(num_threads(), 4);
        set_num_threads(1);
    }

    #[test]
    fn zero_clamps_to_one() {
        let _guard = lock();
        set_num_threads(0);
        assert_eq!(num_threads(), 1);
        set_num_threads(1);
    }
}
