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
echo "2. Building Rust extension with static HDF5..."
echo "   (This tests that HDF5 compiles from source correctly)"

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
if grep -q "before-script-linux:" .github/workflows/publish.yml; then
    echo "   ⚠️  WARNING: before-script-linux is still present in publish.yml"
    echo "      This is unnecessary with static HDF5 and may cause CI failures."
    echo "      Consider removing it with: grep -A2 before-script-linux .github/workflows/publish.yml"
else
    echo "   ✓ No unnecessary before-script-linux in publish.yml"
fi

echo ""
echo "============================================"
echo "✅ All checks passed! Safe to push to GitHub"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Commit the workflow fix: git add .github/workflows/publish.yml"
echo "  2. Push to main and create a release tag"
echo "  3. GitHub Actions will build wheels for all platforms"
echo ""
