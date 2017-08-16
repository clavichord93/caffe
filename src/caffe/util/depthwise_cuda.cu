#include <algorithm>

#include "caffe/common.hpp"
#include "caffe/util/depthwise_cuda.hpp"

namespace caffe {

template <typename Dtype>
__global__ void depthwise_forward_gpu_cuda_kernel(const int n,
    const Dtype* data_in, const Dtype* weight, Dtype* data_out,
    const int height, const int width, const int multiplier, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int height_out, const int width_out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index = index / width_out;
    const int h_out = h_index % height_out;
    const int w_out = index % width_out;
    const int c_in = h_index / height_out;
    const int c_out = c_in * multiplier;
    const int h_offset = h_out * stride_h - pad_h;
    const int w_offset = w_out * stride_w - pad_w;
    Dtype* data_out_ptr = data_out;
    data_out_ptr += (c_out * height_out + h_out) * width_out + w_out;
    const Dtype* data_in_ptr = data_in;
    data_in_ptr += (c_in * height + h_offset) * width + w_offset;
    const Dtype* weight_ptr = weight;
    weight += c_out * kernel_h * kernel_w;
    for (int k = 0; k < multiplier; ++k) {
      Dtype val = 0;
      for (int i = 0; i < kernel_h; ++i) {
        for (int j = 0; j < kernel_w; ++j) {
          int h_in = h_offset + i * dilation_h;
          int w_in = w_offset + j * dilation_w;
          if (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) {
            val += data_in_ptr[i * dilation_h * width + j * dilation_w] *
                weight_ptr[i * kernel_w + j];
          }
        }
      }
      *data_out_ptr = val;
      data_out_ptr += height_out * width_out;
      weight_ptr += kernel_h * kernel_w;
    }
  }
}

template <typename Dtype>
void depthwise_forward_gpu_cuda(const Dtype* data_in, const Dtype* weight,
    Dtype* data_out, const int channels, const int height, const int width,
    const int multiplier, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w) {
  int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
      stride_h + 1;
  int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
      stride_w + 1;
  int num_kernels = channels * height_out * width_out;
  depthwise_forward_gpu_cuda_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), 
                                              CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_in, weight, data_out, height, width, multiplier,
      kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
      dilation_w, height_out, width_out);
  CUDA_POST_KERNEL_CHECK;
}

template void depthwise_forward_gpu_cuda<float>(const float* data_in,
    const float* weight, float* data_out, const int channels, const int height,
    const int width, const int multiplier, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w);
template void depthwise_forward_gpu_cuda<double>(const double* data_in,
    const double* weight, double* data_out, const int channels,
    const int height, const int width, const int multiplier, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w);

template <typename Dtype>
__global__ void depthwise_backward_gpu_cuda_kernel(const int n,
    const Dtype* data_out, const Dtype* weight, Dtype* data_in,
    const int height, const int width, const int multiplier, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int height_out, const int width_out) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    const int w_in = index % width + pad_w;
    const int h_in = (index / width) % height + pad_h;
    const int c_in = index / (width * height);
    const int c_out = c_in * multiplier;
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_out_start =
        (w_in < kernel_extent_w) ? 0 : (w_in - kernel_extent_w) / stride_w + 1;
    const int w_out_end = min(w_in / stride_w + 1, width_out);
    const int h_out_start =
        (h_in < kernel_extent_h) ? 0 : (h_in - kernel_extent_h) / stride_h + 1;
    const int h_out_end = min(h_in / stride_h + 1, height_out);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    const Dtype* data_out_ptr = data_out;
    data_out_ptr += c_out * height_out * width_out;
    const Dtype* weight_ptr = weight;
    weight_ptr += c_out * kernel_h * kernel_w;
    for (int k = 0; k < multiplier; ++k) {
      for (int h_out = h_out_start; h_out < h_out_end; h_out += 1) {
        for (int w_out = w_out_start; w_out < w_out_end; w_out += 1) {
          int h_k = (h_in - h_out * stride_h);
          int w_k = (w_in - w_out * stride_w);
          if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
            h_k /= dilation_h;
            w_k /= dilation_w;
            val += data_out_ptr[h_out * width_out + w_out] *
                weight_ptr[h_k * kernel_w + w_k];
          }
        }
      }
      data_out_ptr += height_out * width_out;
      weight_ptr += kernel_h * kernel_w;
    }
    data_in[index] = val;
  }
}

template <typename Dtype>
void depthwise_backward_gpu_cuda(const Dtype* data_out, const Dtype* weight,
    Dtype* data_in, const int channels, const int height, const int width,
    const int multiplier, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w) {
  int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
      stride_h + 1;
  int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
      stride_w + 1;
  int num_kernels = channels * height * width;
  depthwise_backward_gpu_cuda_kernel<Dtype>
      <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_out, weight, data_in, height, width, multiplier,
      kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
      dilation_w, height_out, width_out);
  CUDA_POST_KERNEL_CHECK;
}

template void depthwise_backward_gpu_cuda<float>(const float* data_out,
    const float* weight, float* data_in, const int channels, const int height,
    const int width, const int multiplier, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w);
template void depthwise_backward_gpu_cuda<double>(const double* data_out,
    const double* weight, double* data_in, const int channels, const int height,
    const int width, const int multiplier, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w);

}  // namespace caffe
