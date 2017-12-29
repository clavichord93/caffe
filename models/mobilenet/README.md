# MobileNet prototxt

MobileNet prototxt using different implementations for depthwise convolutions.

1. `mobilenet-channel-by-channel.prototxt`
  * Original implementation in Caffe;
  * Apply standard convolution to each channel.
2. `mobilenet-specialized-kernel.prototxt`
  * A specialized CUDA kernel function for depthwise convolutions;
3. `mobilenet-diagonalwise-refactorization.prototxt`
  * Implement depthwise convolutions by converting the convolutional kernels to a diagonal matrix;
