#!/usr/bin/env bash
# Build COUNT's native table builder and host-resident lookup library.
#
#   ./count_native/build.sh            -> count_native/count_build, count_native/libcount_lookup.so
#
# Needs a C++17 compiler with OpenMP (g++ or clang; -fopenmp falls back to single-threaded
# lookups if the toolchain has none). No third-party dependencies.
set -euo pipefail

cd "$(dirname "$0")/.."          # repo root: the binaries live next to the count/ python package
HERE=count_native
mkdir -p "$HERE"

CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--O3 -march=native -std=c++17 -fopenmp -Wall -Wextra}"

echo "count_native: building with $CXX ($CXXFLAGS)"

# The builder is a one-shot CLI; the lookup library is loaded by count/table.py via ctypes.
$CXX $CXXFLAGS -o "$HERE/count_build" "$HERE/count_build.cpp"
$CXX $CXXFLAGS -fPIC -shared -o "$HERE/libcount_lookup.so" "$HERE/count_lookup.cpp"

echo "count_native: built $HERE/count_build and $HERE/libcount_lookup.so"
