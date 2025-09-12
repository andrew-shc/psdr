# PSDR

![I](assets/I.gif)

![I_grad](assets/I_grad.gif)

## Compilation Steps

```sh
rm -rf build  # for hard reset
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release  # initial setup
cmake --build build -j"$(nproc)"  # rebuild

./build/bin/optixPathTracer
```