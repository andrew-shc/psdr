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
#include <optix.h>

#include "optixPathTracer.h"
#include "random.h"

#include <sutil/vec_math.h>
#include <cuda/helpers.h>

// for intellisense
#include <optix_device.h>

extern "C"
{
    __constant__ Params params;
}

//------------------------------------------------------------------------------
//
// Orthonormal basis helper
//
//------------------------------------------------------------------------------

struct Onb
{
    __forceinline__ __device__ Onb(const float3 &normal)
    {
        m_normal = normal;

        if (fabs(m_normal.x) > fabs(m_normal.z))
        {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0;
        }
        else
        {
            m_binormal.x = 0;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross(m_binormal, m_normal);
    }

    __forceinline__ __device__ void inverse_transform(float3 &p) const
    {
        p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};

//------------------------------------------------------------------------------
//
// Utility functions
//
//------------------------------------------------------------------------------

static __forceinline__ __device__ RadiancePRD loadClosesthitRadiancePRD()
{
    RadiancePRD prd = {};

    prd.attenuation.x = __uint_as_float(optixGetPayload_0());
    prd.attenuation.y = __uint_as_float(optixGetPayload_1());
    prd.attenuation.z = __uint_as_float(optixGetPayload_2());
    prd.seed = optixGetPayload_3();
    prd.depth = optixGetPayload_4();
    // >> prd.pdf_throughput
    // >> prd.pdf_radiance
    // >> prd.gradient_throughput
    // >> prd.gradient_radiance
    // >> prd.num_params_hit
    prd.pdf_throughput = __uint_as_float(optixGetPayload_5());
    prd.pdf_radiance = __uint_as_float(optixGetPayload_6());
    prd.gradient_throughput.x = __uint_as_float(optixGetPayload_7());
    prd.gradient_throughput.y = __uint_as_float(optixGetPayload_8());
    prd.gradient_throughput.z = __uint_as_float(optixGetPayload_9());
    prd.gradient_radiance.x = __uint_as_float(optixGetPayload_10());
    prd.gradient_radiance.y = __uint_as_float(optixGetPayload_11());
    prd.gradient_radiance.z = __uint_as_float(optixGetPayload_12());
    prd.num_params_hit = optixGetPayload_13();

    return prd;
}

static __forceinline__ __device__ RadiancePRD loadMissRadiancePRD()
{
    RadiancePRD prd = {};
    return prd;
}

static __forceinline__ __device__ void storeClosesthitRadiancePRD(RadiancePRD prd)
{
    optixSetPayload_0(__float_as_uint(prd.attenuation.x));
    optixSetPayload_1(__float_as_uint(prd.attenuation.y));
    optixSetPayload_2(__float_as_uint(prd.attenuation.z));

    optixSetPayload_3(prd.seed);
    optixSetPayload_4(prd.depth);

    // >> prd.pdf_throughput
    // >> prd.pdf_radiance
    // >> prd.gradient_throughput
    // >> prd.gradient_radiance
    // >> prd.num_params_hit
    optixSetPayload_5(__float_as_uint(prd.pdf_throughput));
    optixSetPayload_6(__float_as_uint(prd.pdf_radiance));
    optixSetPayload_7(__float_as_uint(prd.gradient_throughput.x));
    optixSetPayload_8(__float_as_uint(prd.gradient_throughput.y));
    optixSetPayload_9(__float_as_uint(prd.gradient_throughput.z));
    optixSetPayload_10(__float_as_uint(prd.gradient_radiance.x));
    optixSetPayload_11(__float_as_uint(prd.gradient_radiance.y));
    optixSetPayload_12(__float_as_uint(prd.gradient_radiance.z));
    optixSetPayload_13(prd.num_params_hit);

    optixSetPayload_14(__float_as_uint(prd.emitted.x));
    optixSetPayload_15(__float_as_uint(prd.emitted.y));
    optixSetPayload_16(__float_as_uint(prd.emitted.z));

    optixSetPayload_17(__float_as_uint(prd.radiance.x));
    optixSetPayload_18(__float_as_uint(prd.radiance.y));
    optixSetPayload_19(__float_as_uint(prd.radiance.z));

    optixSetPayload_20(__float_as_uint(prd.origin.x));
    optixSetPayload_21(__float_as_uint(prd.origin.y));
    optixSetPayload_22(__float_as_uint(prd.origin.z));

    optixSetPayload_23(__float_as_uint(prd.direction.x));
    optixSetPayload_24(__float_as_uint(prd.direction.y));
    optixSetPayload_25(__float_as_uint(prd.direction.z));

    optixSetPayload_26(prd.done);
}

static __forceinline__ __device__ void storeMissRadiancePRD(RadiancePRD prd)
{
    optixSetPayload_14(__float_as_uint(prd.emitted.x));
    optixSetPayload_15(__float_as_uint(prd.emitted.y));
    optixSetPayload_16(__float_as_uint(prd.emitted.z));

    optixSetPayload_17(__float_as_uint(prd.radiance.x));
    optixSetPayload_18(__float_as_uint(prd.radiance.y));
    optixSetPayload_19(__float_as_uint(prd.radiance.z));

    optixSetPayload_26(prd.done);
}

static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3 &p)
{
    // Uniformly sample disk.
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // Project up to hemisphere.
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}

static __forceinline__ __device__ void basic_sample_hemisphere(const float u1, const float u2, float3 &p)
{
    const float z = u1; // uniform in [0,1]
    const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);
    p.z = z;
}

static __forceinline__ __device__ void traceRadiance(
    OptixTraversableHandle handle,
    float3 ray_origin,
    float3 ray_direction,
    float tmin,
    float tmax,
    RadiancePRD &prd)
{
    unsigned int u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18, u19, u20, u21, u22, u23, u24, u25, u26;

    u0 = __float_as_uint(prd.attenuation.x);
    u1 = __float_as_uint(prd.attenuation.y);
    u2 = __float_as_uint(prd.attenuation.z);
    u3 = prd.seed;
    u4 = prd.depth;
    u5 = __float_as_uint(prd.pdf_throughput);
    u6 = __float_as_uint(prd.pdf_radiance);
    u7 = __float_as_uint(prd.gradient_throughput.x);
    u8 = __float_as_uint(prd.gradient_throughput.y);
    u9 = __float_as_uint(prd.gradient_throughput.z);
    u10 = __float_as_uint(prd.gradient_radiance.x);
    u11 = __float_as_uint(prd.gradient_radiance.y);
    u12 = __float_as_uint(prd.gradient_radiance.z);
    u13 = prd.num_params_hit;

    // Note:
    // This demonstrates the usage of the OptiX shader execution reordering
    // (SER) API.  In the case of this computationally simple shading code,
    // there is no real performance benefit.  However, with more complex shaders
    // the potential performance gains offered by reordering are significant.
    optixTraverse(
        PAYLOAD_TYPE_RADIANCE,
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f, // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0,              // SBT offset
        RAY_TYPE_COUNT, // SBT stride
        0,              // missSBTIndex
        u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18, u19, u20, u21, u22, u23, u24, u25, u26);
    optixReorder(
        // Application specific coherence hints could be passed in here
    );

    optixInvoke(
        PAYLOAD_TYPE_RADIANCE,
        u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18, u19, u20, u21, u22, u23, u24, u25, u26);

    prd.attenuation = make_float3(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2));
    prd.seed = u3;
    prd.depth = u4;
    prd.pdf_throughput = __uint_as_float(u5);
    prd.pdf_radiance = __uint_as_float(u6);
    prd.gradient_throughput = make_float3(__uint_as_float(u7), __uint_as_float(u8), __uint_as_float(u9));
    prd.gradient_radiance = make_float3(__uint_as_float(u10), __uint_as_float(u11), __uint_as_float(u12));
    prd.num_params_hit = u13;

    prd.emitted = make_float3(__uint_as_float(u14), __uint_as_float(u15), __uint_as_float(u16));
    prd.radiance = make_float3(__uint_as_float(u17), __uint_as_float(u18), __uint_as_float(u19));
    prd.origin = make_float3(__uint_as_float(u20), __uint_as_float(u21), __uint_as_float(u22));
    prd.direction = make_float3(__uint_as_float(u23), __uint_as_float(u24), __uint_as_float(u25));
    prd.done = u26;
}

// Returns true if ray is occluded, else false
static __forceinline__ __device__ bool traceOcclusion(
    OptixTraversableHandle handle,
    float3 ray_origin,
    float3 ray_direction,
    float tmin,
    float tmax)
{
    // We are only casting probe rays so no shader invocation is needed
    optixTraverse(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax, 0.0f, // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,              // SBT offset
        RAY_TYPE_COUNT, // SBT stride
        0               // missSBTIndex
    );
    return optixHitObjectIsHit();
}

//------------------------------------------------------------------------------
//
// Programs
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__rg()
{
    const int w = params.width;
    const int h = params.height;
    const float3 eye = params.eye;
    const float3 U = params.U;
    const float3 V = params.V;
    const float3 W = params.W;
    const float3 W_normalized = normalize(W); // Normalized camera forward direction
    const uint3 idx = optixGetLaunchIndex();
    const int launch_seed = params.launch_seed;

    unsigned int seed = tea<4>(idx.y * w + idx.x, launch_seed);

    float3 result = make_float3(0.0f);
    float3 result_grads = make_float3(0.0f);
    int i = params.samples_per_launch;
    do
    {
        // The center of each pixel is at fraction (0.5,0.5)
        const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));

        const float2 d = 2.0f * make_float2(
                                    (static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(w),
                                    (static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(h)) -
                         1.0f;
        float3 ray_direction = normalize(d.x * U + d.y * V + W);
        float3 ray_origin = eye;

        // Cosine term between ray direction and camera forward direction
        float cos_vignette = dot(ray_direction, W_normalized);

        RadiancePRD prd;
        prd.attenuation = make_float3(1.f) * cos_vignette;
        prd.seed = seed;
        prd.depth = 0;
        prd.pdf_throughput = 1.0f;
        prd.pdf_radiance = 1.0f;
        prd.gradient_throughput = make_float3(1.f) * cos_vignette;
        prd.gradient_radiance = make_float3(1.f);
        prd.num_params_hit = 0;

        for (;;)
        {
            traceRadiance(
                params.handle,
                ray_origin,
                ray_direction,
                0.01f, // tmin       // TODO: smarter offset
                1e16f, // tmax
                prd);

            result += prd.emitted;
            result += prd.radiance * prd.attenuation / (prd.pdf_radiance * prd.pdf_throughput);

            float r = 0.0f;
            float g = 0.0f;
            float b = 0.0f;
            if (prd.num_params_hit > 1)
            {
                r = powf(params.parameter.x, prd.num_params_hit - 1);
                g = powf(params.parameter.y, prd.num_params_hit - 1);
                b = powf(params.parameter.z, prd.num_params_hit - 1);
            }
            else if (prd.num_params_hit == 1)
            {
                r = 1.0f;
                g = 1.0f;
                b = 1.0f;
            }
            result_grads += (prd.gradient_throughput * prd.gradient_radiance) / (prd.pdf_radiance * prd.pdf_throughput) * prd.num_params_hit * make_float3(r, g, b);

            // Russian roulette using attenuation magnitude as survival probability
            const float survival_prob = length(prd.attenuation) * 0.9;
            const bool russian_roulette_terminate = rnd(prd.seed) > survival_prob;
            const bool done = prd.done || russian_roulette_terminate;

            if (done)
                break;

            // Compensate for Russian roulette
            prd.pdf_throughput *= survival_prob;

            ray_origin = prd.origin;
            ray_direction = prd.direction;

            ++prd.depth;
        }
    } while (--i);

    const uint3 launch_index = optixGetLaunchIndex();
    const unsigned int image_index = launch_index.y * params.width + launch_index.x;
    float3 accum_color = result / static_cast<float>(params.samples_per_launch);

    float3 accum_gradients = result_grads / static_cast<float>(params.samples_per_launch);

    // subframe index is used when you're sampling again and again to get better result over time
    // different to what we want

    // if (subframe_index > 0)
    // {
    //     const float a = 1.0f / static_cast<float>(subframe_index + 1);
    //     const float3 accum_color_prev = make_float3(params.accum_buffer[image_index]);
    //     accum_color = lerp(accum_color_prev, accum_color, a);
    // }
    params.accum_buffer[image_index] = make_float4(accum_color, 1.0f);
    params.frame_buffer[image_index] = make_color(accum_color);
    params.gradient_buffer[image_index] = make_color(accum_gradients);

    params.frame_buffer_radiance[image_index] = make_float4(accum_color, 1.0f);
    params.gradient_buffer_radiance[image_index] = make_float4(accum_gradients, 1.0f);
}

extern "C" __global__ void __miss__radiance()
{
    optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);

    MissData *rt_data = reinterpret_cast<MissData *>(optixGetSbtDataPointer());
    RadiancePRD prd = loadMissRadiancePRD();

    prd.radiance = make_float3(rt_data->bg_color); // keep 0; not sure if PDFs are needed for this
    prd.emitted = make_float3(0.f);
    prd.done = true;

    storeMissRadiancePRD(prd);
}

extern "C" __global__ void __closesthit__radiance()
{
    optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);

    HitGroupData *rt_data = (HitGroupData *)optixGetSbtDataPointer();

    const int prim_idx = optixGetPrimitiveIndex();
    const float3 ray_dir = optixGetWorldRayDirection();
    const int vert_idx_offset = prim_idx * 3;

    const float3 v0 = make_float3(rt_data->vertices[vert_idx_offset + 0]);
    const float3 v1 = make_float3(rt_data->vertices[vert_idx_offset + 1]);
    const float3 v2 = make_float3(rt_data->vertices[vert_idx_offset + 2]);
    const bool is_parameter = rt_data->is_parameter;
    const float3 N_0 = normalize(cross(v1 - v0, v2 - v0));

    const float3 N = faceforward(N_0, -ray_dir, N_0);
    const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

    RadiancePRD prd = loadClosesthitRadiancePRD();

    if (is_parameter)
    {
        prd.num_params_hit += 1;
    }

    if (prd.depth == 0)
        prd.emitted = rt_data->emission_color;
    else
        prd.emitted = make_float3(0.0f);

    unsigned int seed = prd.seed;
    {
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);

        float3 w_in;
        // basic_sample_hemisphere(z1, z2, w_in);
        cosine_sample_hemisphere(z1, z2, w_in);
        Onb onb(N);
        onb.inverse_transform(w_in);
        prd.direction = w_in;
        prd.origin = P;

        // float cos_theta = dot(w_in, N);
        // float3 f = rt_data->diffuse_color / M_PIf * cos_theta;
        float3 f = rt_data->diffuse_color;
        prd.attenuation *= f;
        prd.pdf_throughput *= 1.0; // cosine-weighted sampling PDF cancels with BRDF*cos

        if (!is_parameter)
            prd.gradient_throughput *= f;
    }

    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    prd.seed = seed;

    ParallelogramLight light = params.light;

    // light sampling only when the final depth is reached OR light is accessible

    const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

    // Calculate properties of light sample (for area based pdf)
    const float Ldist = length(light_pos - P);
    const float3 L = normalize(light_pos - P);
    const float nDl = dot(N, L);
    const float LnDl = -dot(light.normal, L);

    const float A = length(cross(light.v1, light.v2));
    const bool occluded =
        traceOcclusion(
            params.handle,
            P,
            L,
            0.01f,          // tmin
            Ldist - 0.01f); // tmax

    //           MAX DEPTH
    // if (prd.depth >= 20 || !occluded)

    prd.radiance = make_float3(0.0f);
    prd.gradient_radiance = make_float3(0.0f);
    prd.pdf_radiance = 1.0f;
    if (!occluded && (nDl > 0.0f && LnDl > 0.0f))
    {

        float G = nDl * LnDl / (M_PIf * Ldist * Ldist);

        float3 LeG = light.emission * G;
        prd.radiance = LeG;
        prd.gradient_radiance = LeG; // if (!is_parameter) // assume the light is never the parameter (for now)
        prd.pdf_radiance *= 1.0f / A;
    }

    prd.done = false;

    storeClosesthitRadiancePRD(prd);
}
