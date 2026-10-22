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
