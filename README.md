# hip-on-nv - HIP on nv platforms

Inludes inline and library mode. This is created to work with [PyHIP](https://github.com/jatinx/PyHIP) so that one python wrapper works seamlessly with both platforms.

Make sure you use clang to build it.

**Not ready for production. Still lot of work to be done, increase API coverage and complete error enums**.

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
