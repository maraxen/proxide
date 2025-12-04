import sys
import unittest
from unittest import mock

import numpy as np
import pytest
from grain._src.python import options as grain_options
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import prefetch

from priox.ops import prefetch as priox_prefetch


class MockIterDataset(dataset.IterDataset):
    def __init__(self, elements):
        self._elements = elements
        self._parents = []

    def __iter__(self):
        return iter(self._elements)

    @property
    def parents(self):
        return self._parents

def test_get_element_size_bytes():
    # Primitives
    assert priox_prefetch._get_element_size_bytes(1) == sys.getsizeof(1)
    assert priox_prefetch._get_element_size_bytes(1.0) == sys.getsizeof(1.0)
    assert priox_prefetch._get_element_size_bytes("a") == sys.getsizeof("a")

    # Numpy
    arr = np.zeros((10, 10), dtype=np.float32)
    assert priox_prefetch._get_element_size_bytes(arr) == arr.nbytes

    # Lists/Tuples
    l_list = [1, 2, 3]
    # Recursive size: sum of elements.
    # Note: _get_element_size_bytes adds up recursive calls.
    # It does NOT add the size of the container itself in the list case
    # (it iterates and adds child sizes).
    # Wait, let's check code:
    # elif isinstance(element, list | tuple):
    #   for item in element:
    #     size += _get_element_size_bytes(item)
    # It seems it IGNORES the container overhead for list/tuple/dict, only counting contents?
    # No, let's re-read:
    # size = 0
    # if isinstance(element, dict): ... size += recursive
    # else: size += sys.getsizeof(element) (for others)
    # So for list/dict, it ONLY counts values deep size? That seems to be the implementation.

    expected = sum(sys.getsizeof(x) for x in l_list)
    assert priox_prefetch._get_element_size_bytes(l_list) == expected

    t = (1, 2, 3)
    expected = sum(sys.getsizeof(x) for x in t)
    assert priox_prefetch._get_element_size_bytes(t) == expected

    # Dict
    d = {"a": 1, "b": 2}
    expected = sum(sys.getsizeof(x) for x in d.values())
    assert priox_prefetch._get_element_size_bytes(d) == expected

    # Nested
    nested = {"a": [np.zeros(10), np.zeros(10)]}
    expected = 2 * np.zeros(10).nbytes
    assert priox_prefetch._get_element_size_bytes(nested) == expected

def test_get_average_element_size_mb():
    # 1 MB array
    element_size = 1024 * 1024
    # Create an object that has this size approximately (numpy array)
    arr = np.zeros(element_size, dtype=np.uint8)

    ds = MockIterDataset([arr] * 10)

    # Check 1 sample
    avg_mb = priox_prefetch._get_average_element_size_mb([ds], samples_to_check=1)
    # Should be close to 1.0 MB
    assert np.isclose(avg_mb, 1.0, atol=0.01)

    # Empty dataset
    ds_empty = MockIterDataset([])
    avg_mb = priox_prefetch._get_average_element_size_mb([ds_empty], samples_to_check=1)
    assert avg_mb == 0

def test_get_num_workers():
    # Mock _get_average_element_size_mb
    with mock.patch("priox.ops.prefetch._get_average_element_size_mb") as mock_avg:
        mock_avg.return_value = 100.0 # 100 MB

        ds = MockIterDataset([])

        # Budget 1000 MB -> 10 workers
        workers = priox_prefetch._get_num_workers(ds, ram_budget_mb=1000, max_workers=20)
        assert workers == 10

        # Budget 200 MB -> 2 workers
        workers = priox_prefetch._get_num_workers(ds, ram_budget_mb=200, max_workers=20)
        assert workers == 2

        # Cap at max_workers
        workers = priox_prefetch._get_num_workers(ds, ram_budget_mb=10000, max_workers=5)
        assert workers == 5

        # Zero size -> max workers
        mock_avg.return_value = 0
        workers = priox_prefetch._get_num_workers(ds, ram_budget_mb=1000, max_workers=10)
        assert workers == 10

def test_get_buffer_size():
    # Mock _get_average_element_size_mb and _find_prefetch_iter_dataset_parents
    with mock.patch("priox.ops.prefetch._get_average_element_size_mb") as mock_avg:
        with mock.patch("priox.ops.prefetch._find_prefetch_iter_dataset_parents") as mock_parents:
            mock_avg.return_value = 10.0 # 10 MB
            ds = MockIterDataset([])
            # Assume 1 parent found
            mock_parents.return_value = [ds]

            # Budget 100 MB -> buffer size 10
            opts = priox_prefetch._get_buffer_size(ds, ram_budget_mb=100, max_buffer_size=100)
            assert opts.prefetch_buffer_size == 10

            # Cap at max_buffer_size
            opts = priox_prefetch._get_buffer_size(ds, ram_budget_mb=10000, max_buffer_size=50)
            assert opts.prefetch_buffer_size == 50

            # Zero size -> max buffer
            mock_avg.return_value = 0
            opts = priox_prefetch._get_buffer_size(ds, ram_budget_mb=100, max_buffer_size=20)
            assert opts.prefetch_buffer_size == 20

def test_pick_performance_config():
    with mock.patch("priox.ops.prefetch._get_num_workers") as mock_workers:
        with mock.patch("priox.ops.prefetch._get_buffer_size") as mock_buffer:
            mock_workers.return_value = 4
            mock_buffer.return_value = grain_options.ReadOptions(prefetch_buffer_size=10)

            ds = MockIterDataset([])
            config = priox_prefetch.pick_performance_config(
                ds, ram_budget_mb=1000, max_workers=8, max_buffer_size=100
            )

            assert config.multiprocessing_options.num_workers == 4
            assert config.read_options.prefetch_buffer_size == 10

def test_find_prefetch_iter_dataset_parents():
    ds_root = MockIterDataset([1, 2])

    try:
        # Create a PrefetchIterDataset.
        # PrefetchIterDataset(parent, read_options, multiprocessing_options)
        pf_ds = prefetch.PrefetchIterDataset(
            ds_root,
            read_options=grain_options.ReadOptions()
        )

        # Test 1: Direct PrefetchDataset
        parents = priox_prefetch._find_prefetch_iter_dataset_parents(pf_ds)
        # Should return list containing parent of pf_ds, which is ds_root
        assert len(parents) == 1
        assert parents[0] is ds_root

        # Test 2: Wrapped in another dataset
        ds_wrapper = MockIterDataset([])
        ds_wrapper._parents = [pf_ds]

        parents = priox_prefetch._find_prefetch_iter_dataset_parents(ds_wrapper)
        assert len(parents) == 1
        assert parents[0] is ds_root

        # Test 3: Multiple branches?
        # If we have two parents
        ds_wrapper2 = MockIterDataset([])
        pf_ds2 = prefetch.PrefetchIterDataset(
            ds_root,
            read_options=grain_options.ReadOptions()
        )
        ds_wrapper2._parents = [pf_ds, pf_ds2]
        parents = priox_prefetch._find_prefetch_iter_dataset_parents(ds_wrapper2)
        assert len(parents) == 2
        assert parents[0] is ds_root
        assert parents[1] is ds_root

    except Exception as e:
        pytest.fail(f"Failed to instantiate PrefetchIterDataset or run test: {e}")
