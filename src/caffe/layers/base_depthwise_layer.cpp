#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/depthwise_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BaseDepthwiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  multiplier_ = this->layer_param_.convolution_param().multiplier();
  num_output_ = channels_ * multiplier_;
  CHECK_GT(num_output_, 0);
  conv_out_channels_ = num_output_;
  conv_in_channels_ = channels_;
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BaseDepthwiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * channels_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    col_buffer_shape_.push_back(output_shape_[i]);
  }
  col_buffer_.Reshape(col_buffer_shape_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BaseDepthwiseLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output) {
  conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
  const Dtype* col_buff = col_buffer_.cpu_data();
  caffe_cpu_gemm_batched<Dtype>(CblasNoTrans, CblasNoTrans,
      multiplier_, conv_out_spatial_dim_, kernel_dim_,
      (Dtype)1., weights, col_buff, (Dtype)0., output, channels_);
}

template <typename Dtype>
void BaseDepthwiseLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseDepthwiseLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  caffe_cpu_gemm_batched<Dtype>(CblasTrans, CblasNoTrans,
      kernel_dim_, conv_out_spatial_dim_, multiplier_,
      (Dtype)1., weights, output, (Dtype)0., col_buff, channels_);
  conv_col2im_cpu(col_buff, input);
}

template <typename Dtype>
void BaseDepthwiseLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
  const Dtype* col_buff = col_buffer_.cpu_data();
  caffe_cpu_gemm_batched<Dtype>(CblasNoTrans, CblasTrans,
      multiplier_, kernel_dim_, conv_out_spatial_dim_,
      (Dtype)1., output, col_buff, (Dtype)1., weights, channels_);
}

template <typename Dtype>
void BaseDepthwiseLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseDepthwiseLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output) {
  conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
  const Dtype* col_buff = col_buffer_.gpu_data();
  caffe_gpu_gemm_batched<Dtype>(CblasNoTrans, CblasNoTrans,
      multiplier_, conv_out_spatial_dim_, kernel_dim_,
      (Dtype)1., weights, col_buff, (Dtype)0., output, channels_);
}

template <typename Dtype>
void BaseDepthwiseLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseDepthwiseLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  caffe_gpu_gemm_batched<Dtype>(CblasTrans, CblasNoTrans,
      kernel_dim_, conv_out_spatial_dim_, multiplier_,
      (Dtype)1., weights, output, (Dtype)0., col_buff, channels_);
  conv_col2im_gpu(col_buff, input);
}

template <typename Dtype>
void BaseDepthwiseLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
  const Dtype* col_buff = col_buffer_.gpu_data();
  caffe_gpu_gemm_batched<Dtype>(CblasNoTrans, CblasTrans,
      multiplier_, kernel_dim_, conv_out_spatial_dim_,
      (Dtype)1., output, col_buff, (Dtype)1., weights, channels_);
}

template <typename Dtype>
void BaseDepthwiseLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

template <typename Dtype>
void BaseDepthwiseLayer<Dtype>::forward_gpu_cuda(const Dtype* data_in,
    const Dtype* weight, Dtype* data_out) {
  // Single-Sample depth-wise implementation
  // const Dtype* data_in_ptr = data_in;
  // Dtype* data_out_ptr = data_out;
  // for (int i = 0; i < num_; ++i) {
  //   depthwise_forward_gpu_cuda(data_in_ptr, weight, data_out_ptr, 1, channels_,
  //       conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
  //       multiplier_, kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
  //       pad_.cpu_data()[0], pad_.cpu_data()[1], stride_.cpu_data()[0],
  //       stride_.cpu_data()[1], dilation_.cpu_data()[0], dilation_.cpu_data()[1]);
  //   data_in_ptr += bottom_dim_;
  //   data_out_ptr += top_dim_;
  // }
  // Multi-Sample depth-wise implementation
  depthwise_forward_gpu_cuda(data_in, weight, data_out, num_, channels_,
      conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
      multiplier_, kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
      pad_.cpu_data()[0], pad_.cpu_data()[1], stride_.cpu_data()[0],
      stride_.cpu_data()[1], dilation_.cpu_data()[0], dilation_.cpu_data()[1]);
}

template <typename Dtype>
void BaseDepthwiseLayer<Dtype>::backward_gpu_cuda(const Dtype* data_out,
    const Dtype* weight, Dtype* data_in) {
  // Single-Sample depth-wise implementation
  // Dtype* data_in_ptr = data_in;
  // const Dtype* data_out_ptr = data_out;
  // for (int i = 0; i < num_; ++i) {
  //   depthwise_backward_data_gpu_cuda(data_out_ptr, weight, data_in_ptr, 1, channels_,
  //       conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
  //       multiplier_, kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
  //       pad_.cpu_data()[0], pad_.cpu_data()[1], stride_.cpu_data()[0],
  //       stride_.cpu_data()[1], dilation_.cpu_data()[0], dilation_.cpu_data()[1]);
  //   data_in_ptr += bottom_dim_;
  //   data_out_ptr += top_dim_;
  // }
  // Multi-Sample depth-wise implementation
  depthwise_backward_data_gpu_cuda(data_out, weight, data_in, num_, channels_,
      conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
      multiplier_, kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
      pad_.cpu_data()[0], pad_.cpu_data()[1], stride_.cpu_data()[0],
      stride_.cpu_data()[1], dilation_.cpu_data()[0], dilation_.cpu_data()[1]);
}

template <typename Dtype>
void BaseDepthwiseLayer<Dtype>::weight_gpu_cuda(const Dtype* data_out,
    const Dtype* data_in, Dtype* weight) {
  // Single-Sample depth-wise implementation
  // const Dtype* data_in_ptr = data_in;
  // const Dtype* data_out_ptr = data_out;
  // for (int i = 0; i < num_; ++i) {
  //   depthwise_backward_filter_gpu_cuda(data_out_ptr, data_in_ptr, weight, 1, channels_,
  //       conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
  //       multiplier_, kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
  //       pad_.cpu_data()[0], pad_.cpu_data()[1], stride_.cpu_data()[0],
  //       stride_.cpu_data()[1], dilation_.cpu_data()[0], dilation_.cpu_data()[1]);
  //   data_in_ptr += bottom_dim_;
  //   data_out_ptr += top_dim_;
  // }
  // Multi-Sample depth-wise implementation
  depthwise_backward_filter_gpu_cuda(data_out, data_in, weight, num_, channels_,
      conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
      multiplier_, kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
      pad_.cpu_data()[0], pad_.cpu_data()[1], stride_.cpu_data()[0],
      stride_.cpu_data()[1], dilation_.cpu_data()[0], dilation_.cpu_data()[1]);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseDepthwiseLayer);

}  // namespace caffe
