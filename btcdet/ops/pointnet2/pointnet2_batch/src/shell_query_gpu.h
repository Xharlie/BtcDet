#ifndef _SHELL_QUERY_GPU_H
#define _SHELL_QUERY_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int shell_query_wrapper_fast(int b, int n, int m, float inner_radius, float outer_radius, int nsample,
	at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor);

void shell_query_kernel_launcher_fast(int b, int n, int m, float inner_radius, float outer_radius, int nsample,
	const float *xyz, const float *new_xyz, int *idx);

#endif
