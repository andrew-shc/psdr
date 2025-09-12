/*

 * SPDX-FileCopyrightText: Copyright (c) 2019 - 2024  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <optix_stack_size.h>

#include <GLFW/glfw3.h>

#include "optixPathTracer.h"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

bool resize_dirty = false;
bool minimized = false;

// Camera state
bool camera_changed = true;
sutil::Camera camera;
sutil::Trackball trackball;

// Mouse state
int32_t mouse_button = -1;

//------------------------------------------------------------------------------
//
// Local types
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

struct RenderResult
{
    std::vector<uchar4> color_data;
    std::vector<uchar4> gradient_data;
    std::vector<float4> color_radiance_data;
    std::vector<float4> gradient_radiance_data;
    int width;
    int height;

    // Helper functions to create ImageBuffer objects for saving
    sutil::ImageBuffer getColorBuffer() const
    {
        sutil::ImageBuffer buffer;
        buffer.data = const_cast<void *>(static_cast<const void *>(color_data.data()));
        buffer.width = width;
        buffer.height = height;
        buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
        return buffer;
    }

    sutil::ImageBuffer getGradientBuffer() const
    {
        sutil::ImageBuffer buffer;
        buffer.data = const_cast<void *>(static_cast<const void *>(gradient_data.data()));
        buffer.width = width;
        buffer.height = height;
        buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
        return buffer;
    }

    sutil::ImageBuffer getColorRadianceBuffer() const
    {
        sutil::ImageBuffer buffer;
        buffer.data = const_cast<void *>(static_cast<const void *>(color_radiance_data.data()));
        buffer.width = width;
        buffer.height = height;
        buffer.pixel_format = sutil::BufferImageFormat::FLOAT4;
        return buffer;
    }

    sutil::ImageBuffer getGradientRadianceBuffer() const
    {
        sutil::ImageBuffer buffer;
        buffer.data = const_cast<void *>(static_cast<const void *>(gradient_radiance_data.data()));
        buffer.width = width;
        buffer.height = height;
        buffer.pixel_format = sutil::BufferImageFormat::FLOAT4;
        return buffer;
    }
};

template <typename T>
struct Record
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<RayGenData> RayGenRecord;
typedef Record<MissData> MissRecord;
typedef Record<HitGroupData> HitGroupRecord;

struct Vertex
{
    float x, y, z, pad;
};

struct IndexedTriangle
{
    uint32_t v1, v2, v3, pad;
};

struct Instance
{
    float transform[12];
};

struct PathTracerState
{
    OptixDeviceContext context = 0;

    OptixTraversableHandle gas_handle = 0; // Traversable handle for triangle AS
    CUdeviceptr d_gas_output_buffer = 0;   // Triangle AS memory
    CUdeviceptr d_vertices = 0;

    OptixModule ptx_module = 0;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipeline pipeline = 0;

    OptixProgramGroup raygen_prog_group = 0;
    OptixProgramGroup radiance_miss_group = 0;
    OptixProgramGroup radiance_hit_group = 0;

    CUstream stream = 0;
    Params params;
    Params *d_params;

    OptixShaderBindingTable sbt = {};
};

//------------------------------------------------------------------------------
//
// Scene data
//
//------------------------------------------------------------------------------

const int32_t TRIANGLE_COUNT = 32;
const int32_t MAT_COUNT = 5;

const static std::array<Vertex, TRIANGLE_COUNT * 3> g_vertices =
    {{// Floor  -- white lambert
      {0.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 559.2f, 0.0f},
      {556.0f, 0.0f, 559.2f, 0.0f},
      {0.0f, 0.0f, 0.0f, 0.0f},
      {556.0f, 0.0f, 559.2f, 0.0f},
      {556.0f, 0.0f, 0.0f, 0.0f},

      // Ceiling -- white lambert
      {0.0f, 548.8f, 0.0f, 0.0f},
      {556.0f, 548.8f, 0.0f, 0.0f},
      {556.0f, 548.8f, 559.2f, 0.0f},

      {0.0f, 548.8f, 0.0f, 0.0f},
      {556.0f, 548.8f, 559.2f, 0.0f},
      {0.0f, 548.8f, 559.2f, 0.0f},

      // Back wall -- white lambert
      {0.0f, 0.0f, 559.2f, 0.0f},
      {0.0f, 548.8f, 559.2f, 0.0f},
      {556.0f, 548.8f, 559.2f, 0.0f},

      {0.0f, 0.0f, 559.2f, 0.0f},
      {556.0f, 548.8f, 559.2f, 0.0f},
      {556.0f, 0.0f, 559.2f, 0.0f},

      // Right wall -- green lambert
      {0.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 548.8f, 0.0f, 0.0f},
      {0.0f, 548.8f, 559.2f, 0.0f},

      {0.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 548.8f, 559.2f, 0.0f},
      {0.0f, 0.0f, 559.2f, 0.0f},

      // Left wall -- red lambert
      {556.0f, 0.0f, 0.0f, 0.0f},
      {556.0f, 0.0f, 559.2f, 0.0f},
      {556.0f, 548.8f, 559.2f, 0.0f},

      {556.0f, 0.0f, 0.0f, 0.0f},
      {556.0f, 548.8f, 559.2f, 0.0f},
      {556.0f, 548.8f, 0.0f, 0.0f},

      // Short block -- white lambert
      {130.0f, 165.0f, 65.0f, 0.0f},
      {82.0f, 165.0f, 225.0f, 0.0f},
      {242.0f, 165.0f, 274.0f, 0.0f},

      {130.0f, 165.0f, 65.0f, 0.0f},
      {242.0f, 165.0f, 274.0f, 0.0f},
      {290.0f, 165.0f, 114.0f, 0.0f},

      {290.0f, 0.0f, 114.0f, 0.0f},
      {290.0f, 165.0f, 114.0f, 0.0f},
      {240.0f, 165.0f, 272.0f, 0.0f},

      {290.0f, 0.0f, 114.0f, 0.0f},
      {240.0f, 165.0f, 272.0f, 0.0f},
      {240.0f, 0.0f, 272.0f, 0.0f},

      {130.0f, 0.0f, 65.0f, 0.0f},
      {130.0f, 165.0f, 65.0f, 0.0f},
      {290.0f, 165.0f, 114.0f, 0.0f},

      {130.0f, 0.0f, 65.0f, 0.0f},
      {290.0f, 165.0f, 114.0f, 0.0f},
      {290.0f, 0.0f, 114.0f, 0.0f},

      {82.0f, 0.0f, 225.0f, 0.0f},
      {82.0f, 165.0f, 225.0f, 0.0f},
      {130.0f, 165.0f, 65.0f, 0.0f},

      {82.0f, 0.0f, 225.0f, 0.0f},
      {130.0f, 165.0f, 65.0f, 0.0f},
      {130.0f, 0.0f, 65.0f, 0.0f},

      {240.0f, 0.0f, 272.0f, 0.0f},
      {240.0f, 165.0f, 272.0f, 0.0f},
      {82.0f, 165.0f, 225.0f, 0.0f},

      {240.0f, 0.0f, 272.0f, 0.0f},
      {82.0f, 165.0f, 225.0f, 0.0f},
      {82.0f, 0.0f, 225.0f, 0.0f},

      // Tall block -- white lambert
      {423.0f, 330.0f, 247.0f, 0.0f},
      {265.0f, 330.0f, 296.0f, 0.0f},
      {314.0f, 330.0f, 455.0f, 0.0f},

      {423.0f, 330.0f, 247.0f, 0.0f},
      {314.0f, 330.0f, 455.0f, 0.0f},
      {472.0f, 330.0f, 406.0f, 0.0f},

      {423.0f, 0.0f, 247.0f, 0.0f},
      {423.0f, 330.0f, 247.0f, 0.0f},
      {472.0f, 330.0f, 406.0f, 0.0f},

      {423.0f, 0.0f, 247.0f, 0.0f},
      {472.0f, 330.0f, 406.0f, 0.0f},
      {472.0f, 0.0f, 406.0f, 0.0f},

      {472.0f, 0.0f, 406.0f, 0.0f},
      {472.0f, 330.0f, 406.0f, 0.0f},
      {314.0f, 330.0f, 456.0f, 0.0f},

      {472.0f, 0.0f, 406.0f, 0.0f},
      {314.0f, 330.0f, 456.0f, 0.0f},
      {314.0f, 0.0f, 456.0f, 0.0f},

      {314.0f, 0.0f, 456.0f, 0.0f},
      {314.0f, 330.0f, 456.0f, 0.0f},
      {265.0f, 330.0f, 296.0f, 0.0f},

      {314.0f, 0.0f, 456.0f, 0.0f},
      {265.0f, 330.0f, 296.0f, 0.0f},
      {265.0f, 0.0f, 296.0f, 0.0f},

      {265.0f, 0.0f, 296.0f, 0.0f},
      {265.0f, 330.0f, 296.0f, 0.0f},
      {423.0f, 330.0f, 247.0f, 0.0f},

      {265.0f, 0.0f, 296.0f, 0.0f},
      {423.0f, 330.0f, 247.0f, 0.0f},
      {423.0f, 0.0f, 247.0f, 0.0f},

      // Ceiling light -- emmissive
      {343.0f, 548.6f, 227.0f, 0.0f},
      {213.0f, 548.6f, 227.0f, 0.0f},
      {213.0f, 548.6f, 332.0f, 0.0f},

      {343.0f, 548.6f, 227.0f, 0.0f},
      {213.0f, 548.6f, 332.0f, 0.0f},
      {343.0f, 548.6f, 332.0f, 0.0f}}};

static std::array<uint32_t, TRIANGLE_COUNT> g_mat_indices = {{
    0, 0,                         // Floor         -- white lambert
    0, 0,                         // Ceiling       -- white lambert
    0, 0,                         // Back wall     -- white lambert
    1, 1,                         // Right wall    -- green lambert
    2, 2,                         // Left wall     -- red lambert
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // Short block   -- white lambert
    4, 4, 0, 0, 0, 0, 0, 0, 0, 0, // Tall block    -- white lambert
    //             ^         ----- back side
    3, 3 // Ceiling light -- emmissive
}};

const std::array<float3, MAT_COUNT> g_emission_colors =
    {{
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 0.33333f}, //   {1.0f, 1.0f, 0.33333f} //  {15.0f, 15.0f, 5.0f},
        {0.0f, 0.0f, 0.0f},
    }};

const std::array<float3, MAT_COUNT> g_diffuse_colors_gt =
    {{{0.80f, 0.80f, 0.80f},
      {0.05f, 0.80f, 0.05f},
      {0.80f, 0.05f, 0.05f},
      {0.50f, 0.00f, 0.00f},
      {0.10f, 0.30f, 0.95f}}};

std::array<float3, MAT_COUNT> g_diffuse_colors_init =
    {{{0.80f, 0.80f, 0.80f},
      {0.05f, 0.80f, 0.05f},
      {0.80f, 0.05f, 0.05f}, // {0.50f, 0.50f, 0.50f},
      {0.50f, 0.00f, 0.00f},
      {0.50f, 0.50f, 0.50f}}};

const std::array<bool, MAT_COUNT> g_material_parameter_mask =
    {{false,
      false,
      false,
      false,
      true}};

//------------------------------------------------------------------------------
//
// Helper functions
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

void printUsageAndExit(const char *argv0)
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "         --launch-samples | -s       Number of samples per pixel per launch (default 16)\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 768x768\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit(0);
}

void initLaunchParams(PathTracerState &state, int32_t samples_per_launch)
{
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&state.params.accum_buffer),
        state.params.width * state.params.height * sizeof(float4)));
    state.params.frame_buffer = nullptr;             // Will be set when output buffer is mapped
    state.params.gradient_buffer = nullptr;          // Will be set when output buffer is mapped
    state.params.frame_buffer_radiance = nullptr;    // Will be set when output buffer is mapped
    state.params.gradient_buffer_radiance = nullptr; // Will be set when output buffer is mapped

    state.params.samples_per_launch = samples_per_launch;
    state.params.launch_seed = 0u;

    state.params.light.emission = make_float3(15.0f, 15.0f, 5.0f);
    state.params.light.corner = make_float3(343.0f, 548.5f, 227.0f);
    state.params.light.v1 = make_float3(0.0f, 0.0f, 105.0f);
    state.params.light.v2 = make_float3(-130.0f, 0.0f, 0.0f);
    state.params.light.normal = normalize(cross(state.params.light.v1, state.params.light.v2));
    state.params.handle = state.gas_handle;

    for (size_t i = 0; i < g_material_parameter_mask.size(); ++i)
    {
        if (g_material_parameter_mask[i])
        {
            state.params.parameter = g_diffuse_colors_init[i];
            break;
        }
    }

    CUDA_CHECK(cudaStreamCreate(&state.stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_params), sizeof(Params)));
}

void handleCameraUpdate(Params &params)
{
    if (!camera_changed)
        return;
    camera_changed = false;

    camera.setAspectRatio(static_cast<float>(params.width) / static_cast<float>(params.height));
    params.eye = camera.eye();
    camera.UVWFrame(params.U, params.V, params.W);
}

void handleResize(sutil::CUDAOutputBuffer<uchar4> &output_buffer, Params &params)
{
    if (!resize_dirty)
        return;
    resize_dirty = false;

    output_buffer.resize(params.width, params.height);

    // Realloc accumulation buffer
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(params.accum_buffer)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&params.accum_buffer),
        params.width * params.height * sizeof(float4)));
}

void updateState(sutil::CUDAOutputBuffer<uchar4> &output_buffer, Params &params)
{
    // Update params on device
    if (camera_changed || resize_dirty)
        params.launch_seed = 0;

    handleCameraUpdate(params);
    handleResize(output_buffer, params);
}

void launchSubframe(
    sutil::CUDAOutputBuffer<uchar4> &output_buffer,
    sutil::CUDAOutputBuffer<uchar4> &gradient_buffer,
    sutil::CUDAOutputBuffer<float4> &color_buffer_radiance,
    sutil::CUDAOutputBuffer<float4> &gradient_buffer_radiance,
    PathTracerState &state)
{
    // Launch
    uchar4 *result_buffer_data = output_buffer.map();
    uchar4 *gradient_buffer_data = gradient_buffer.map();
    float4 *radiance_buffer_data = color_buffer_radiance.map();
    float4 *gradient_radiance_buffer_data = gradient_buffer_radiance.map();
    state.params.frame_buffer = result_buffer_data;
    state.params.gradient_buffer = gradient_buffer_data;
    state.params.frame_buffer_radiance = radiance_buffer_data;
    state.params.gradient_buffer_radiance = gradient_radiance_buffer_data;

    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void *>(state.d_params),
        &state.params, sizeof(Params),
        cudaMemcpyHostToDevice, state.stream));

    OPTIX_CHECK(optixLaunch(
        state.pipeline,
        state.stream,
        reinterpret_cast<CUdeviceptr>(state.d_params),
        sizeof(Params),
        &state.sbt,
        state.params.width,  // launch width
        state.params.height, // launch height
        1                    // launch depth
        ));
    output_buffer.unmap();
    gradient_buffer.unmap();
    color_buffer_radiance.unmap();
    gradient_buffer_radiance.unmap();
    CUDA_SYNC_CHECK();
}

static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

void initCameraState()
{
    camera.setEye(make_float3(278.0f, 273.0f, -900.0f));
    camera.setLookat(make_float3(278.0f, 273.0f, 330.0f));
    camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    camera.setFovY(35.0f);
    camera_changed = true;

    trackball.setCamera(&camera);
    trackball.setMoveSpeed(10.0f);
    trackball.setReferenceFrame(
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 1.0f),
        make_float3(0.0f, 1.0f, 0.0f));
    trackball.setGimbalLock(true);
}

void createContext(PathTracerState &state)
{
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    OptixDeviceContext context;
    CUcontext cu_ctx = 0; // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
#ifdef DEBUG
    // This may incur significant performance cost and should only be done during development.
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));

    state.context = context;
}

void buildMeshAccel(PathTracerState &state)
{
    //
    // copy mesh data to device
    //
    const size_t vertices_size_in_bytes = g_vertices.size() * sizeof(Vertex);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_vertices), vertices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(state.d_vertices),
        g_vertices.data(), vertices_size_in_bytes,
        cudaMemcpyHostToDevice));

    CUdeviceptr d_mat_indices = 0;
    const size_t mat_indices_size_in_bytes = g_mat_indices.size() * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_mat_indices), mat_indices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_mat_indices),
        g_mat_indices.data(),
        mat_indices_size_in_bytes,
        cudaMemcpyHostToDevice));

    //
    // Build triangle GAS
    //
    uint32_t triangle_input_flags[MAT_COUNT] = // One per SBT record for this build input
        {
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};

    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(Vertex);
    triangle_input.triangleArray.numVertices = static_cast<uint32_t>(g_vertices.size());
    triangle_input.triangleArray.vertexBuffers = &state.d_vertices;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = MAT_COUNT;
    triangle_input.triangleArray.sbtIndexOffsetBuffer = d_mat_indices;
    triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &triangle_input,
        1, // num_build_inputs
        &gas_buffer_sizes));

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_buffer_temp_output_gas_and_compacted_size),
        compactedSizeOffset + 8));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char *)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(
        state.context,
        0, // CUDA stream
        &accel_options,
        &triangle_input,
        1, // num build inputs
        d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &state.gas_handle,
        &emitProperty, // emitted property list
        1              // num emitted properties
        ));

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_mat_indices)));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void *)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_gas_output_buffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(state.context, 0, state.gas_handle, state.d_gas_output_buffer, compacted_gas_size, &state.gas_handle));

        CUDA_CHECK(cudaFree((void *)d_buffer_temp_output_gas_and_compacted_size));
    }
    else
    {
        state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

void createModule(PathTracerState &state)
{
    OptixPayloadType payloadType = {};
    // radiance prd
    payloadType.numPayloadValues = sizeof(radiancePayloadSemantics) / sizeof(radiancePayloadSemantics[0]);
    payloadType.payloadSemantics = radiancePayloadSemantics;

    OptixModuleCompileOptions module_compile_options = {};
#if OPTIX_DEBUG_DEVICE_CODE
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
    module_compile_options.numPayloadTypes = 1;
    module_compile_options.payloadTypes = &payloadType;

    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues = 0;
    state.pipeline_compile_options.numAttributeValues = 2;
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    size_t inputSize = 0;
    const char *input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixPathTracer.cu", inputSize);

    OPTIX_CHECK_LOG(optixModuleCreate(
        state.context,
        &module_compile_options,
        &state.pipeline_compile_options,
        input,
        inputSize,
        LOG, &LOG_SIZE,
        &state.ptx_module));
}

void createProgramGroups(PathTracerState &state)
{
    OptixProgramGroupOptions program_group_options = {};

    {
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = state.ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, &raygen_prog_group_desc,
            1, // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &state.raygen_prog_group));
    }

    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = state.ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, &miss_prog_group_desc,
            1, // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &state.radiance_miss_group));
    }

    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &hit_prog_group_desc,
            1, // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &state.radiance_hit_group));
    }
}

void createPipeline(PathTracerState &state)
{
    OptixProgramGroup program_groups[] =
        {
            state.raygen_prog_group,
            state.radiance_miss_group,
            state.radiance_hit_group,
        };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2;

    OPTIX_CHECK_LOG(optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        LOG, &LOG_SIZE,
        &state.pipeline));

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.raygen_prog_group, &stack_sizes, state.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.radiance_miss_group, &stack_sizes, state.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.radiance_hit_group, &stack_sizes, state.pipeline));

    uint32_t max_trace_depth = 2;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        max_cc_depth,
        max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size));

    const uint32_t max_traversal_depth = 1;
    OPTIX_CHECK(optixPipelineSetStackSize(
        state.pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversal_depth));
}

void createSBT(
    PathTracerState &state,
    const std::array<float3, MAT_COUNT> &d_emission_colors,
    const std::array<float3, MAT_COUNT> &d_diffuse_colors,
    const std::array<bool, MAT_COUNT> &d_material_parameter_mask)
{
    CUdeviceptr d_raygen_record;
    const size_t raygen_record_size = sizeof(RayGenRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_raygen_record), raygen_record_size));

    RayGenRecord rg_sbt = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice));

    CUdeviceptr d_miss_records;
    const size_t miss_record_size = sizeof(MissRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_miss_records), miss_record_size * RAY_TYPE_COUNT));

    MissRecord ms_sbt[1];
    OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_miss_group, &ms_sbt[0]));
    ms_sbt[0].data.bg_color = make_float4(0.0f);

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_miss_records),
        ms_sbt,
        miss_record_size * RAY_TYPE_COUNT,
        cudaMemcpyHostToDevice));

    CUdeviceptr d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof(HitGroupRecord);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_hitgroup_records),
        hitgroup_record_size * RAY_TYPE_COUNT * MAT_COUNT));

    HitGroupRecord hitgroup_records[RAY_TYPE_COUNT * MAT_COUNT];
    for (int i = 0; i < MAT_COUNT; ++i)
    {
        {
            const int sbt_idx = i * RAY_TYPE_COUNT + 0; // SBT for radiance ray-type for ith material

            OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_hit_group, &hitgroup_records[sbt_idx]));
            hitgroup_records[sbt_idx].data.emission_color = d_emission_colors[i];
            hitgroup_records[sbt_idx].data.diffuse_color = d_diffuse_colors[i];
            hitgroup_records[sbt_idx].data.is_parameter = d_material_parameter_mask[i];
            hitgroup_records[sbt_idx].data.vertices = reinterpret_cast<float4 *>(state.d_vertices);
        }

        // Note that we do not need to use any program groups for occlusion
        // rays as they are traced as 'probe rays' with no shading.
    }

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_hitgroup_records),
        hitgroup_records,
        hitgroup_record_size * RAY_TYPE_COUNT * MAT_COUNT,
        cudaMemcpyHostToDevice));

    state.sbt.raygenRecord = d_raygen_record;
    state.sbt.missRecordBase = d_miss_records;
    state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
    state.sbt.missRecordCount = RAY_TYPE_COUNT;
    state.sbt.hitgroupRecordBase = d_hitgroup_records;
    state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
    state.sbt.hitgroupRecordCount = RAY_TYPE_COUNT * MAT_COUNT;
}

RenderResult render(PathTracerState &state, sutil::CUDAOutputBufferType output_buffer_type, int seed = 0)
{
    sutil::CUDAOutputBuffer<uchar4> output_buffer(output_buffer_type, state.params.width, state.params.height);
    sutil::CUDAOutputBuffer<uchar4> gradient_buffer(output_buffer_type, state.params.width, state.params.height);
    sutil::CUDAOutputBuffer<float4> color_buffer_radiance(output_buffer_type, state.params.width, state.params.height);
    sutil::CUDAOutputBuffer<float4> gradient_buffer_radiance(output_buffer_type, state.params.width, state.params.height);

    state.params.launch_seed = seed;

    handleCameraUpdate(state.params);
    handleResize(output_buffer, state.params);
    handleResize(gradient_buffer, state.params);
    launchSubframe(output_buffer, gradient_buffer, color_buffer_radiance, gradient_buffer_radiance, state);

    // Copy data to avoid dangling pointers
    RenderResult result;
    result.width = output_buffer.width();
    result.height = output_buffer.height();

    const size_t pixel_count = result.width * result.height;

    // Copy uchar4 color data
    result.color_data.resize(pixel_count);
    std::memcpy(result.color_data.data(), output_buffer.getHostPointer(), pixel_count * sizeof(uchar4));

    // Copy uchar4 gradient data
    result.gradient_data.resize(pixel_count);
    std::memcpy(result.gradient_data.data(), gradient_buffer.getHostPointer(), pixel_count * sizeof(uchar4));

    // Copy float4 color radiance data
    result.color_radiance_data.resize(pixel_count);
    std::memcpy(result.color_radiance_data.data(), color_buffer_radiance.getHostPointer(), pixel_count * sizeof(float4));

    // Copy float4 gradient radiance data
    result.gradient_radiance_data.resize(pixel_count);
    std::memcpy(result.gradient_radiance_data.data(), gradient_buffer_radiance.getHostPointer(), pixel_count * sizeof(float4));

    return result;
}

void cleanupState(PathTracerState &state)
{
    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_miss_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_hit_group));
    OPTIX_CHECK(optixModuleDestroy(state.ptx_module));
    OPTIX_CHECK(optixDeviceContextDestroy(state.context));

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_vertices)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_gas_output_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.params.accum_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_params)));
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    PathTracerState state;
    state.params.width = 1920;
    state.params.height = 1080;
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;

    int ITERATIONS = 100;
    float LEARNING_RATE = 0.1f;
    int32_t SAMPLES_PER_LAUNCH = 16;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--launch-samples" || arg == "-s")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            SAMPLES_PER_LAUNCH = atoi(argv[++i]);
        }
        else if (arg == "--iterations" || arg == "-n")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            ITERATIONS = atoi(argv[++i]);
        }
        else if (arg == "--learning-rate" || arg == "-l")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            LEARNING_RATE = static_cast<float>(atof(argv[++i]));
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit(argv[0]);
        }
    }

    //
    // Parse command line options
    //

    // for (int i = 1; i < argc; ++i)
    // {
    //     const std::string arg = argv[i];
    //     if (arg == "--launch-samples" || arg == "-s")
    //     {
    //         if (i >= argc - 1)
    //             printUsageAndExit(argv[0]);
    //         samples_per_launch = atoi(argv[++i]);
    //     }
    //     else
    //     {
    //         std::cerr << "Unknown option '" << argv[i] << "'\n";
    //         printUsageAndExit(argv[0]);
    //     }
    // }

    try
    {
        initCameraState();

        //
        // Set up OptiX state
        //
        createContext(state);
        buildMeshAccel(state);
        createModule(state);
        createProgramGroups(state);
        createPipeline(state);

        createSBT(state, g_emission_colors, g_diffuse_colors_gt, g_material_parameter_mask);

        // Clear the loss file at the start of the program
        std::ofstream loss_file_clear("misc/output/loss.txt", std::ios::trunc);
        if (loss_file_clear.is_open())
        {
            loss_file_clear.close();
        }

        // Generate ground truth
        initLaunchParams(state, 128);
        auto gt_result = render(state, output_buffer_type, 0);

        std::string outfile("misc/output/I_gt.png");
        sutil::saveImage(outfile.c_str(), gt_result.getColorBuffer(), false);
        std::string outfile_grad("misc/output/I_gt_grad.png");
        sutil::saveImage(outfile_grad.c_str(), gt_result.getGradientBuffer(), false);

        for (int i = 0; i < ITERATIONS; i++)
        {
            createSBT(state, g_emission_colors, g_diffuse_colors_init, g_material_parameter_mask);
            initLaunchParams(state, 32);
            auto result = render(state, output_buffer_type, i * 2 + 0);
            auto result_grad = render(state, output_buffer_type, i * 2 + 1); // needs independent sampling

            float gradient_r = 0.0f;
            float gradient_g = 0.0f;
            float gradient_b = 0.0f;
            float loss = 0.0f;

            for (unsigned int idx = 0; idx < state.params.width * state.params.height; ++idx)
            {
                // Use high-precision float4 radiance buffers instead of quantized uint8 buffers
                float4 color_theta = result.color_radiance_data[idx];
                float4 color_init = gt_result.color_radiance_data[idx];
                float4 grad_data = result_grad.gradient_radiance_data[idx];

                float r_theta = color_theta.x;
                float g_theta = color_theta.y;
                float b_theta = color_theta.z;
                float r_init = color_init.x;
                float g_init = color_init.y;
                float b_init = color_init.z;
                float r_grad = grad_data.x;
                float g_grad = grad_data.y;
                float b_grad = grad_data.z;

                gradient_r += (r_theta - r_init) * r_grad;
                gradient_g += (g_theta - g_init) * g_grad;
                gradient_b += (b_theta - b_init) * b_grad;
                loss += (r_theta - r_init) * (r_theta - r_init);
                loss += (g_theta - g_init) * (g_theta - g_init);
                loss += (b_theta - b_init) * (b_theta - b_init);
            }

            std::cout << "Iteration " << i << " - Loss: " << loss
                      << " | Gradient R: " << gradient_r
                      << " G: " << gradient_g
                      << " B: " << gradient_b << std::endl;

            // Save loss to text file
            std::ofstream loss_file("misc/output/loss.txt", std::ios::app);
            if (loss_file.is_open())
            {
                loss_file << i << " " << loss << std::endl;
                loss_file.close();
            }

            for (size_t j = 0; j < g_material_parameter_mask.size(); ++j)
            {
                if (g_material_parameter_mask[j])
                {
                    g_diffuse_colors_init[j].x -= LEARNING_RATE * gradient_r / (state.params.width * state.params.height);
                    g_diffuse_colors_init[j].y -= LEARNING_RATE * gradient_g / (state.params.width * state.params.height);
                    g_diffuse_colors_init[j].z -= LEARNING_RATE * gradient_b / (state.params.width * state.params.height);
                    g_diffuse_colors_init[j].x = std::max(0.0f, std::min(1.0f, g_diffuse_colors_init[j].x));
                    g_diffuse_colors_init[j].y = std::max(0.0f, std::min(1.0f, g_diffuse_colors_init[j].y));
                    g_diffuse_colors_init[j].z = std::max(0.0f, std::min(1.0f, g_diffuse_colors_init[j].z));
                    std::cout << "Updated parameters to: "
                              << g_diffuse_colors_init[j].x << ", "
                              << g_diffuse_colors_init[j].y << ", "
                              << g_diffuse_colors_init[j].z << std::endl;

                    float mse = ((g_diffuse_colors_init[j].x - g_diffuse_colors_gt[j].x) * (g_diffuse_colors_init[j].x - g_diffuse_colors_gt[j].x) +
                                 (g_diffuse_colors_init[j].y - g_diffuse_colors_gt[j].y) * (g_diffuse_colors_init[j].y - g_diffuse_colors_gt[j].y) +
                                 (g_diffuse_colors_init[j].z - g_diffuse_colors_gt[j].z) * (g_diffuse_colors_init[j].z - g_diffuse_colors_gt[j].z)) /
                                3.0f;
                    std::cout << "Albedo MSE vs GT: " << mse << std::endl;

                    break;
                }
            }

            std::string outfile("misc/output/I_" + std::to_string(i) + ".png");
            sutil::saveImage(outfile.c_str(), result.getColorBuffer(), false);
            std::string outfile_grad("misc/output/I_grad_" + std::to_string(i) + ".png");
            sutil::saveImage(outfile_grad.c_str(), result.getGradientBuffer(), false);
        }

        cleanupState(state);
    }
    catch (std::exception &e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
