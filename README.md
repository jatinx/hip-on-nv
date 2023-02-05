# hip-on-nv - HIP on nv platforms

Inludes inline and library mode.

## How to build

- `git clone https://github.com/jatinx/hip-on-nv.git && cd hip-on-nv`
- `mkdir build && cd build`
- `cmake .. -DCMAKE_INSTALL_PREFIX=<Install_Dir> -DCMAKE_CXX_COMPILER=clang++`
- `make install`
- `export HIPNV_INSTALL_PATH=<Install_Dir>`

## Using `hipcc`

`bin` folder has script `hipcc` which basically forwards calls to `nvcc`.

Usage: `hipcc file.cu`

Options:

- `--nvhip-help` : Print help for the utility
- `--nvhip-lib-mode` : Link `nvhip64` lib, has unlined symbols of hip.
- `--nvhip-debug-print` : Print debug info of scripts
