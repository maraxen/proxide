#!/bin/bash
# Test local wheel building before pushing to GitHub CI
# This validates that the wheel build process works locally before triggering the GitHub workflow

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "============================================"
echo "Proxide Wheel Build Test"
echo "============================================"
echo ""

# Check prerequisites
echo "1. Checking build prerequisites..."
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust/cargo not found. Install from https://rustup.rs/"
    exit 1
fi
echo "   ✓ cargo found: $(cargo --version)"

if ! command -v rustc &> /dev/null; then
    echo "❌ rustc not found"
    exit 1
fi
echo "   ✓ rustc found: $(rustc --version)"

if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found"
    exit 1
fi
echo "   ✓ python3 found: $(python3 --version)"

echo ""
echo "2. Building Rust extension with static HDF5 (native target)..."
echo "   (This tests that HDF5 compiles from source correctly)"
echo "   Note: aarch64 cross-compilation can't be tested locally without"
echo "   aarch64 target installed. CI will test that."

# Clean previous builds
rm -rf target/release/deps/lib_proxider* 2>/dev/null || true

# Build with release profile
cargo build --release --package proxide_py 2>&1 | grep -E "(Compiling|Finished|error|warning: profiles)" || true

if [ -f "target/release/deps/lib_proxider.so" ]; then
    SIZE=$(ls -lh target/release/deps/lib_proxider.so | awk '{print $5}')
    echo "   ✓ Extension built successfully (size: $SIZE)"
else
    echo "❌ Failed to build extension"
    exit 1
fi

echo ""
echo "3. Checking Cargo.toml for static HDF5 feature..."
if grep -q 'hdf5.*static' Cargo.toml; then
    echo "   ✓ Static HDF5 feature is enabled"
else
    echo "❌ Static HDF5 feature not found in Cargo.toml"
    echo "   Run: grep 'hdf5' Cargo.toml"
    exit 1
fi

echo ""
echo "4. Checking GitHub workflow configuration..."
if grep -q 'CFLAGS_aarch64_unknown_linux_gnu' .github/workflows/publish.yml; then
    echo "   ✓ ARM cross-compile flags configured"
else
    echo "⚠️  WARNING: ARM cross-compile flags not found in publish.yml"
fi

if grep -q 'CXXFLAGS_aarch64_unknown_linux_gnu' .github/workflows/publish.yml; then
    echo "   ✓ ARM C++ cross-compile flags configured"
fi

if grep -q 'before-script-linux:' .github/workflows/publish.yml; then
    if grep -A5 'before-script-linux:' .github/workflows/publish.yml | grep -q 'hdf5'; then
        echo "   ⚠️  WARNING: before-script-linux still has HDF5 install (not needed)"
    else
        echo "   ✓ before-script-linux configured (minimal, no HDF5 install)"
    fi
fi

echo ""
echo "============================================"
echo "✅ All native checks passed!"
echo "============================================"
echo ""
echo "NOTE: Full platform testing happens in CI:"
echo "  • Linux x86_64: Tests native build"
echo "  • Linux aarch64: Tests ARM cross-compile (CFLAGS_aarch64_unknown_linux_gnu)"
echo "  • macOS x86_64/aarch64: Platform-specific builds"
echo "  • Windows x64: Platform-specific build"
echo ""
echo "Next steps:"
echo "  1. Push changes: git push origin"
echo "  2. Tag release: git tag -a vX.Y.Z"
echo "  3. Push tags: git push origin --tags"
echo "  4. Monitor: gh run list --workflow publish.yml"
echo ""

