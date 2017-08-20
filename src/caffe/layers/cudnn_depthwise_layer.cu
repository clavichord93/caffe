#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_depthwise_layer.hpp"

namespace caffe {

__global__ void sync_depthwise() { }

template <typename Dtype>
void CuDNNDepthwiseLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    // Forward through cuDNN.
    // Filters.
    for (int g = 0; g < group_; ++g) {
      CUDNN_CHECK(cudnnConvolutionForward(handle_[0],
          cudnn::dataType<Dtype>::one,
          bottom_descs_[i], bottom_data + bottom_offset_ * g,
          filter_desc_, weight + weight_offset_ * g,
          conv_descs_[i],
          fwd_algo_[i], workspace[0], workspace_fwd_sizes_[i],
          cudnn::dataType<Dtype>::zero,
          top_descs_[i], top_data + top_offset_ * g));
    }

    // Bias.
    if (this->bias_term_) {
      const Dtype* bias_data = this->blobs_[1]->gpu_data();
      CUDNN_CHECK(cudnnAddTensor(handle_[0], cudnn::dataType<Dtype>::one,
          bias_desc_, bias_data, cudnn::dataType<Dtype>::one, top_descs_[i],
          top_data));
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_depthwise<<<1, 1>>>();
  }
}

template <typename Dtype>
void CuDNNDepthwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  const Dtype* mask = NULL;
  int num_weight = 0;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
    mask = mask_.gpu_data();
    num_weight = mask_.count();
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through cuDNN in parallel over gradients.
    // Gradient w.r.t. bias.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0],
          cudnn::dataType<Dtype>::one, top_descs_[i], top_diff,
          cudnn::dataType<Dtype>::one, bias_desc_, bias_diff));
    }

    // Gradient w.r.t. weights.
    if (this->param_propagate_down_[0]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      for (int g = 0; g < group_; ++g) {
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle_[1],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            top_descs_[i], top_diff + top_offset_ * g,
            conv_descs_[i], bwd_filter_algo_[i],
            workspace[1], workspace_bwd_filter_sizes_[i],
            cudnn::dataType<Dtype>::one,
            filter_desc_, weight_diff + weight_offset_ * g));
      }
    }

    // Gradient w.r.t. bottom data.
    if (propagate_down[i]) {
      if (weight == NULL) {
        weight = this->blobs_[0]->gpu_data();
      }
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int g = 0; g < group_; ++g) {
        CUDNN_CHECK(cudnnConvolutionBackwardData(handle_[2],
            cudnn::dataType<Dtype>::one,
            filter_desc_, weight + weight_offset_ * g,
            top_descs_[i], top_diff + top_offset_ * g,
            conv_descs_[i], bwd_data_algo_[i],
            workspace[2], workspace_bwd_data_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_depthwise<<<1, 1>>>();
    if (this->param_propagate_down_[0]) {
      const Dtype* unmasked_diff = this->blobs_[0]->gpu_diff();
      caffe_gpu_mul(num_weight, mask, unmasked_diff, weight_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNDepthwiseLayer);

}  // namespace caffe
#endif
