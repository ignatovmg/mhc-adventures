#!/usr/bin/env bash
set -euo pipefail

# Directories to use
SRC_DIR="$(cd $(dirname "$0") && pwd)"
ENV_DIR="${SRC_DIR}/venv"
NUMPROC=$(nproc)

# Specific commits to checkout
PRODY_VERSION=1.10.11
SBLU_COMMIT=0cdd4ef
LIBMOL2_COMMIT=a2a40df
NGRMIN_COMMIT=52c18802
JANSSON_VERSION=2.12
CHECK_VERSION=0.14.0

# Check that we have cmake
#which cmake || echo "Please install cmake >= 3.12" && exit 1

# Create vars.json
sed "s:#SRC_DIR#:${SRC_DIR}:g" vars.json.tpl > vars.json
sed -i "s:#MIN_EXE#:${ENV_DIR}/bin/minimize:g" vars.json

# load modules
if [[ "$(hostname)" == login* ]]; then
    source "${SRC_DIR}/seawulf_modules.txt"
fi

# Setup conda env
if [ ! -d "${ENV_DIR}" ]; then
    conda env create -f "${SRC_DIR}/conda-env.yml" --prefix "${ENV_DIR}"
else
    conda env update -f "${SRC_DIR}/conda-env.yml" --prefix "${ENV_DIR}"
fi

# Create conda environment in the current directory
set +u  # conda references PS1 variable that is not set in scripts
source activate "${ENV_DIR}"
set -u

# Setting env variables
set +u
export PKG_CONFIG_PATH="${ENV_DIR}/lib/pkgconfig:${PKG_CONFIG_PATH}"
export PATH="${ENV_DIR}/lib:${PATH}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH}"
set -u

if [ '1' ]; then

# Install ProDy
pip install prody==${PRODY_VERSION}

# Install psfgen
cp "${SRC_DIR}/deps/psfgen_1.6.5_Linux-x86_64-multicore" "${ENV_DIR}/bin/psfgen" && chmod ug+x "${ENV_DIR}/bin/psfgen"

# Install sb-lab-utils
rm -rf sb-lab-utils
git clone https://bitbucket.org/bu-structure/sb-lab-utils.git
cd sb-lab-utils
git checkout ${SBLU_COMMIT}
pip install -r requirements/pipeline.txt
python setup.py install
cd ../
rm -rf sb-lab-utils

# Install libjansson
rm -rf "jansson-${JANSSON_VERSION}"
wget "https://github.com/akheron/jansson/archive/v${JANSSON_VERSION}.tar.gz" -O jansson.tar.gz
tar zxf jansson.tar.gz
rm -f jansson.tar.gz
cd "jansson-${JANSSON_VERSION}"
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${ENV_DIR}" -DJANSSON_BUILD_DOCS=OFF ../
make -j"$NUMPROC"
make install
cd ../../
rm -rf "jansson-${JANSSON_VERSION}"

# Install check
rm -rf check-${CHECK_VERSION}
wget https://github.com/libcheck/check/releases/download/${CHECK_VERSION}/check-${CHECK_VERSION}.tar.gz
tar xvf check-${CHECK_VERSION}.tar.gz
rm -f check-${CHECK_VERSION}.tar.gz
cd "check-${CHECK_VERSION}"
# Check.h needs to be removed, this is why https://github.com/libcheck/check/issues/172#issuecomment-588405110
rm -f src/check.h
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX="${ENV_DIR}" -DCMAKE_INSTALL_LIBDIR=lib ../
make -j"$NUMPROC"
make install
cd ../../
rm -rf "check-${CHECK_VERSION}"

# Install Gperf
rm -rf gperf-3.1
wget http://ftp.gnu.org/pub/gnu/gperf/gperf-3.1.tar.gz
tar xvf gperf-3.1.tar.gz
rm -f gperf-3.1.tar.gz
cd  gperf-3.1
./configure --prefix "${ENV_DIR}"
make -j"$NUMPROC"
make install
cd ../
rm -rf gperf-3.1

fi

# Install libmol2
rm -rf libmol2
git clone https://bitbucket.org/bu-structure/libmol2.git
cd libmol2
git checkout ${LIBMOL2_COMMIT}
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON \
    -DUSE_LTO=OFF \
    -DCMAKE_PREFIX_PATH="${ENV_DIR}" \
    -DCMAKE_INSTALL_PREFIX="${ENV_DIR}" \
    -DBUILD_NOE=OFF \
    ..

make -j"$NUMPROC"
make test
make install
cd ../../
rm -rf libmol2

# Install nrgmin
rm -rf nrgmin
git clone https://ignatovmg@bitbucket.org/ignatovmg/nrgmin.git
cd nrgmin
git checkout ${NGRMIN_COMMIT}
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON \
    -DOPENMP=ON \
    -DCMAKE_PREFIX_PATH="${ENV_DIR}" \
    -DNOE=OFF \
    ..
make -j"$NUMPROC"
make test
cp nrgmin nrgmin.omp "${ENV_DIR}/bin"
cd ../../
rm -rf nrgmin

# Install the package itself
pip install -e .