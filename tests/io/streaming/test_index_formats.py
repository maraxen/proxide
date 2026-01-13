
import json
from pathlib import Path
from unittest import mock

import pytest

from proxide.io.streaming.array_record import ArrayRecordDataSource


# Mock the ArrayRecordReader since we don't have actual .array_record files
@pytest.fixture
def mock_reader():
    with mock.patch("proxide.io.streaming.array_record.ArrayRecordReader") as MockReader:
        instance = MockReader.return_value
        instance.num_records.return_value = 10
        yield MockReader

def test_legacy_index_format(tmp_path, mock_reader):
    """Test that legacy integer-based index files are correctly handled."""
    # Create dummy array record file
    ar_path = tmp_path / "test.array_record"
    ar_path.touch()
    
    # Create legacy index file: {"id1": 0, "id2": 1}
    index_path = tmp_path / "legacy.index.json"
    legacy_index = {
        "prot_0": 0,
        "prot_1": 1,
        "prot_2": 2
    }
    with open(index_path, "w") as f:
        json.dump(legacy_index, f)
        
    # Initialize source with "inference" split - legacy format should include all
    ds = ArrayRecordDataSource(ar_path, index_path, split="inference")
    
    assert len(ds) == 3
    assert "prot_0" in ds.index
    assert ds.index["prot_0"] == {"idx": [0], "set": "inference"}

def test_modern_index_format(tmp_path, mock_reader):
    """Test that modern dictionary-based index files are correctly handled."""
    ar_path = tmp_path / "test.array_record"
    ar_path.touch()
    
    index_path = tmp_path / "modern.index.json"
    modern_index = {
        "train_1": {"idx": [0], "set": "train"},
        "valid_1": {"idx": [1], "set": "valid"},
        "train_2": {"idx": [2], "set": "train"}
    }
    with open(index_path, "w") as f:
        json.dump(modern_index, f)
        
    # Test train split
    ds_train = ArrayRecordDataSource(ar_path, index_path, split="train")
    assert len(ds_train) == 2
    assert "train_1" in ds_train.index
    assert "train_2" in ds_train.index
    assert "valid_1" not in ds_train.index
    
    # Test valid split
    ds_valid = ArrayRecordDataSource(ar_path, index_path, split="valid")
    assert len(ds_valid) == 1
    assert "valid_1" in ds_valid.index

def test_custom_filter_fn(tmp_path, mock_reader):
    """Test that generic filter_fn works."""
    ar_path = tmp_path / "test.array_record"
    ar_path.touch()
    
    index_path = tmp_path / "generic.index.json"
    # Extended format with custom tags
    generic_index = {
        "p1": {"idx": [0], "quality": "high", "len": 100},
        "p2": {"idx": [1], "quality": "low", "len": 50},
        "p3": {"idx": [2], "quality": "high", "len": 200}
    }
    with open(index_path, "w") as f:
        json.dump(generic_index, f)
        
    # Filter for high quality
    def high_quality_filter(pid, entry):
        return entry.get("quality") == "high"
        
    ds = ArrayRecordDataSource(
        ar_path, 
        index_path, 
        split="train", # split should be ignored when filter_fn is present
        filter_fn=high_quality_filter
    )
    
    assert len(ds) == 2
    assert "p1" in ds.index
    assert "p3" in ds.index
    assert "p2" not in ds.index

def test_mixed_legacy_normalization(tmp_path, mock_reader):
    """Test normalization helper directly (implicit via init)."""
    ar_path = tmp_path / "test.array_record"
    ar_path.touch()
    
    # Legacy format again to verify default split assignment logic
    index_path = tmp_path / "legacy.index.json"
    legacy_index = {"p1": 5}
    with open(index_path, "w") as f:
        json.dump(legacy_index, f)
        
    ds = ArrayRecordDataSource(ar_path, index_path, split="custom_split")
    # All records included for legacy, and 'set' becomes 'custom_split'
    assert ds.index["p1"]["set"] == "custom_split"
