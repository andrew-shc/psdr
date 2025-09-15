# PSDR

Basic albedo optimization (biased).

![I](assets/I.gif)

![I_grad](assets/I_grad.gif)


Albedo optimization on an occluded surface (grey -> blue on top of the long rectanglular prism). The blue tint reflection on top of the long rectangular prism is kind of hard to see, but it slightly gets more bluer till it reaches to something similiar to the ground truth image.

![I](assets/improved_I.gif)

![I_grad](assets/improved_I_grad.gif)


## Compilation Steps

```sh
rm -rf build  # for hard reset
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release  # initial setup
cmake --build build -j"$(nproc)"  # rebuild

./build/bin/optixPathTracer
./build/bin/optixPathTracer --iterations 200 --learning-rate 0.1 --launch-samples 128
```