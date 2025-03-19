/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

uint32_t getHigherMsb(uint32_t n);

__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present);

__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid);

__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges);

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t scan_size;
		float* depths; //[P,1]
		char* scanning_space;
		bool* clamped; //[P,3], 颜色是否为负
		int* internal_radii; //[P,1]
		float2* means2D; //[P,1], float2
		float* cov3D; //[P,6]
		float4* conic_opacity; //[P,1], float4, packed(2D covariance, opacity)
		float* rgb; //[P,3]
		uint32_t* point_offsets; //[P,1]
		uint32_t* tiles_touched; //[P,1] 覆盖的tiles的数量

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size; // 存储用于排序操作的缓冲区大小
		uint64_t* point_list_keys_unsorted; // 未排序的键列表
		uint64_t* point_list_keys; // 排序后的键列表
		uint32_t* point_list_unsorted; // 未排序的点列表
		uint32_t* point_list; // 排序后的点列表
		char* list_sorting_space; // 用于排序操作的缓冲区

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};