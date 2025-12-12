
import struct
from pathlib import Path

def create_dummy_fragments(output_path: Path):
    with open(output_path, "wb") as f:
        # Header
        f.write(b"FRAG")
        f.write(struct.pack("<I", 1))  # Version 1
        f.write(struct.pack("<I", 0))  # Num entries = 0
    print(f"Created dummy fragments library at {output_path}")

if __name__ == "__main__":
    output = Path("rust_ext/data/fragments.bin")
    output.parent.mkdir(parents=True, exist_ok=True)
    create_dummy_fragments(output)
