#ifndef _CAFFE_UTIL_DEPTHWISE_CUDA_HPP_
#define _CAFFE_UTIL_DEPTHWISE_CUDA_HPP_

namespace caffe {

template <typename Dtype>
void depthwise_forward_gpu_cuda(const Dtype* data_in, const Dtype* weight,
    Dtype* data_out, const int batch, const int channels, const int height,
    const int width, const int multiplier, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w);

template <typename Dtype>
void depthwise_backward_gpu_cuda(const Dtype* data_out, const Dtype* weight,
    Dtype* data_in, const int batch, const int channels, const int height, 
    const int width, const int multiplier, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w);

}  // namespace caffe

#endif  // CAFFE_UTIL_DEPTHWISE_CUDA_HPP_
