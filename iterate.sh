#! /usr/bin/env bash

set -eo pipefail

cd ~/cs6886/proj/hls4ml || exit
pushd . || exit
python3 test_pytorch_conv2dtranspose.py

if ! test -d hls4ml_prj; then
    echo "Error: codegen hls4ml_prj does not exist" >&2
    exit
fi

if ! test -d hls4ml_prj/build; then
    cd hls4ml_prj/build
    cmake -DUSER_INCLUDE_PATHS=/opt/intel/oneapi/compiler/2025.0/opt/oclfpga/include/ ..
    cd -
fi

cd hls4ml_prj/build || exit

source /opt/intel/oneapi/setvars.sh || true

make fpga_emu
cd ..
./build/myproject.fpga_emu
python3 ../sanitycheck.py
cd build || exit
test -f myproject.report && rm myproject.report
make report
cd myproject.report.prj || exit

if test "$1" == "--agent"; then
    popd || exit
    exit
fi

code reports/resources/json
brave reports/report.html
popd || exit
