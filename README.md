# PSDR

## Compilation Steps

```sh
rm -rf build  # for hard reset
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

./build/bin/optixPathTracer
```