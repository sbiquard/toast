#!/bin/bash

# Pass extra cmake options to this script, including
# things like -DCMAKE_INSTALL_PREFIX=/path/to/install, etc.

opts="$@"

cmake \
    -DCMAKE_C_COMPILER="icx" \
    -DCMAKE_CXX_COMPILER="icpx" \
    -DCMAKE_C_FLAGS="-O3 -g -fPIC -pthread" \
    -DCMAKE_CXX_FLAGS="-O3 -g -fPIC -pthread" \
    -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DUSE_MKL=TRUE \
    ${opts} \
    ..
