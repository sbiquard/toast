# Install SuiteSparse - only the pieces we need.

# Variables set by code sourcing this:
# - CC
# - CFLAGS
# - CXX
# - CXXFLAGS
# - FC
# - FCFLAGS
# - FCLIBS
# - OMPFLAGS
# - BLAS_LIBRARIES
# - LAPACK_LIBRARIES
# - PREFIX
# - DEPSDIR (to find patches)
# - MAKEJ
# - STATIC (yes/no)
# - CLEANUP (yes/no)

ssparse_version=7.8.2
ssparse_dir=SuiteSparse-${ssparse_version}
ssparse_pkg=${ssparse_dir}.tar.gz

echo "Fetching SuiteSparse..."

if [ ! -e ${ssparse_pkg} ]; then
    curl -SL https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/v${ssparse_version}.tar.gz -o ${ssparse_pkg}
fi

echo "Building SuiteSparse..."

shr="-DNSTATIC:BOOL=ON -DBLA_STATIC:BOOL=OFF"
if [ "${STATIC}" = "yes" ]; then
    shr="-DNSTATIC:BOOL=OFF -DBLA_STATIC:BOOL=ON"
fi

rm -rf ${ssparse_dir}
tar xzf ${ssparse_pkg} \
    && pushd ${ssparse_dir} >/dev/null 2>&1 \
    && cmake \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_Fortran_COMPILER="${FC}" \
    -DCMAKE_C_FLAGS="${CFLAGS}" \
    -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
    -DCMAKE_Fortran_FLAGS="${FCFLAGS}" \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DCMAKE_INSTALL_PATH="${PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release ${shr} \
    -DBLAS_LIBRARIES="${BLAS_LIBRARIES}" \
    -DLAPACK_LIBRARIES="${LAPACK_LIBRARIES}" ${BLA_OPTIONS} \
    -DSUITESPARSE_ENABLE_PROJECTS="cholmod" \
    -S . -B build \
    && cmake --build build \
    && cmake --install build \
    && popd >/dev/null 2>&1

if [ "x${CLEANUP}" = "xyes" ]; then
    rm -rf ${ssparse_dir}
fi
