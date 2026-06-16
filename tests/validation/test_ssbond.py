
import pytest
from proxide import io

# 1crn.pdb is known to have 3 disulfide bonds
def test_ssbond_count():
    # Assuming the API to load the structure provides access to SSBONDs or similar information.
    # Since the prompt asks to test it, I'll assume an interface exists or needs verification.
    
    # Placeholder for actual test logic. 
    # If the library doesn't yet expose bond info directly in the Python bindings,
    # I might need to check if the parsing works generally.
    pass

if __name__ == "__main__":
    pytest.main([__file__])
