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
 #include <cuda.h>
 #include "device_launch_parameters.h"
 #include <cub/cub.cuh>
 #include <cub/device/device_radix_sort.cuh>
 #include <cooperative_groups.h>
 #include <cooperative_groups/reduce.h>
 namespace cg = cooperative_groups;
 #include "rasterizer_impl.h"
 #include "forward.h"
 #include "auxiliary.h"

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
get_max_contributorCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float2* __restrict__ points_xy_image,
    const float* __restrict__ features,
    const float4* __restrict__ conic_opacity,
    float* __restrict__ final_T,
    uint32_t* __restrict__ n_contrib,
    const float* __restrict__ bg_color,
    int* __restrict__ max_contributor,
    float* __restrict__ max_contribute){
    // Identify current tile and associated min/max pixel range.
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y }; // 当前处理的tile的左上角的像素坐标
    uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) }; // 当前处理的tile的右下角的像素坐标
    uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y }; // 当前处理的像素坐标
    uint32_t pix_id = W * pix.y + pix.x; // 当前处理的像素id
    float2 pixf = { (float)pix.x, (float)pix.y }; // 当前处理的像素坐标

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = pix.x < W&& pix.y < H;
    // Done threads can help with fetching, but don't rasterize
    bool done = !inside;

    // Load start/end range of IDs to process in bit sorted list.
    // 当前处理的tile对应的3D gaussian的起始id和结束id
    uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x; // 还有多少3D gaussian需要处理

    // Allocate storage for batches of collectively fetched data.
    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

    // Initialize helper variables
    float T = 1.0f;
    uint32_t contributor = 0;
    uint32_t last_contributor = 0;
    int32_t max_id = -1;
    float max_weight = 0.0;

    // Iterate over batches until all done or range is complete
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        // End if entire block votes that it is done rasterizing
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE)
            break;

        // Collectively fetch per-Gaussian data from global to shared
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y)
        {
            int coll_id = point_list[range.x + progress]; // 当前处理的3D gaussian的id
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = points_xy_image[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
        }
        block.sync();

        // Iterate over current batch
        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
        {
            // Keep track of current position in range
            contributor++;

            // Resample using conic matrix (cf. "Surface 
            // Splatting" by Zwicker et al., 2001)
            float2 xy = collected_xy[j]; // 当前处理的2D gaussian在图像上的中心点坐标
            float2 d = { xy.x - pixf.x, xy.y - pixf.y }; // 当前处理的2D gaussian的中心点到当前处理的pixel的offset
            float4 con_o = collected_conic_opacity[j]; // 当前处理的2D gaussian的协方差矩阵的逆矩阵以及它的不透明度
            float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y; // 计算高斯分布的强度（或权重），用于确定像素在光栅化过程中的贡献程度
            if (power > 0.0f)
                continue;

            // Eq. (2) from 3D Gaussian splatting paper.
            // Obtain alpha by multiplying with Gaussian opacity
            // and its exponential falloff from mean.
            // Avoid numerical instabilities (see paper appendix). 
            float alpha = min(0.99f, con_o.w * exp(power));
            if (alpha < 1.0f / 255.0f)
                continue;
            float test_T = T * (1 - alpha);
            if (test_T < 0.0001f)
            {
                done = true;
                continue;
            }
            
            if(alpha * T > max_weight){
                max_weight = alpha*T;
                max_id = collected_id[j];
            }

            T = test_T;

            // Keep track of last range entry to update this
            // pixel.
            last_contributor = contributor;
        }
    }

    // All threads that treat valid pixel write out their final
    // rendering data to the frame and auxiliary buffers.
    if (inside)
    {
        final_T[pix_id] = T; // 渲染过程后每个像素的最终透明度或透射率值
        n_contrib[pix_id] = last_contributor; // 最后一个贡献的2D gaussian是谁
        max_contributor[pix_id] = max_id;
        max_contribute[pix_id] = max_weight;
    }
}

void FORWARD::get_max_contributor(
    const dim3 grid, dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int W, int H,
    const float2* means2D,
    const float* colors,
    const float4* conic_opacity,
    float* final_T,
    uint32_t* n_contrib,
    const float* bg_color,
    int* max_contributor,
    float* max_contribute){
    get_max_contributorCUDA<<<grid, block >>> (
        ranges,             // 每个瓦片（tile）在排序后的高斯ID列表中的范围
        point_list,         // 排序后的3D gaussian的id列表
        W, H,               // 图像的宽和高
        means2D,            // 每个2D gaussian在图像上的中心点位置
        colors,             // 每个3D gaussian对应的RGB颜色
        conic_opacity,      // 每个2D gaussian的协方差矩阵的逆矩阵以及它的不透明度
        final_T,            // 渲染过程后每个像素的最终透明度或透射率值
        n_contrib,          // 每个pixel的最后一个贡献的2D gaussian是谁
        bg_color,           // 背景颜色
        max_contributor,
        max_contribute);
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::get_max_contributor(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	int* max_contributor, //output
	float* max_contribute, //output
	int* radii, //output
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P); //每个高斯一个
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1); //分块，每块16x16
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height); //每个像素一个
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered; // 存储所有的2D gaussian总共覆盖了多少个tile
	// 将 geomState.point_offsets 数组中最后一个元素的值复制到主机内存中的变量 num_rendered
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// 将每个3D gaussian的对应的tile index和深度存到point_list_keys_unsorted中
    // 将每个3D gaussian的对应的index（第几个3D gaussian）存到point_list_unsorted中
	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys <<<(P + 255) / 256, 256 >>> (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// 对一个键值对列表进行排序。这里的键值对由 binningState.point_list_keys_unsorted 和 binningState.point_list_unsorted 组成
    // 排序后的结果存储在 binningState.point_list_keys 和 binningState.point_list 中
    // binningState.list_sorting_space 和 binningState.sorting_size 指定了排序操作所需的临时存储空间和其大小
    // num_rendered 是要排序的元素总数。0, 32 + bit 指定了排序的最低位和最高位，这里用于确保排序考虑到了足够的位数，以便正确处理所有的键值对
	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	// 将 imgState.ranges 数组中的所有元素设置为 0
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// 识别每个瓦片（tile）在排序后的高斯ID列表中的范围
    // 目的是确定哪些高斯ID属于哪个瓦片，并记录每个瓦片的开始和结束位置
	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges <<<(num_rendered + 255) / 256, 256 >>> (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::get_max_contributor(
		tile_grid, // 在水平和垂直方向上需要多少个块来覆盖整个渲染区域
		block, // 每个块在 X（水平）和 Y（垂直）方向上的线程数
		imgState.ranges, // 每个瓦片（tile）在排序后的高斯ID列表中的范围
		binningState.point_list, // 排序后的3D gaussian的id列表
		width, height, // 图像的宽和高
		geomState.means2D, // 每个2D gaussian在图像上的中心点位置
		feature_ptr, // 每个3D gaussian对应的RGB颜色
		geomState.conic_opacity, // 每个2D gaussian的协方差矩阵的逆矩阵以及它的不透明度
		imgState.accum_alpha, // 渲染过程后每个像素的最终透明度或透射率值
		imgState.n_contrib, // 每个pixel的最后一个贡献的2D gaussian是谁
		background, // 背景颜色
		max_contributor,
		max_contribute), debug)  // 输出图像

	return num_rendered;
}