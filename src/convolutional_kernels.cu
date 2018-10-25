#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

__global__ void binarize_kernel(float *x, int n, float *binary)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    binary[i] = (x[i] >= 0) ? 1 : -1;
}

void binarize_gpu(float *x, int n, float *binary)
{
    binarize_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, binary);
    check_error(cudaPeekAtLastError());
}

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary)
{
    int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (s >= size) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < n; ++i){
        mean += fabsf(input[i*size + s]);
    }
    mean = mean / n;
    for(i = 0; i < n; ++i){
        binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
    }
}

void binarize_input_gpu(float *input, int n, int size, float *binary)
{
    binarize_input_kernel<<<cuda_gridsize(size), BLOCK>>>(input, n, size, binary);
    check_error(cudaPeekAtLastError());
}


__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary)
{
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < size; ++i){
        mean += fabsf(weights[f*size + i]);
    }
    mean = mean / size;
    for(i = 0; i < size; ++i){
        binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        //binary[f*size + i] = weights[f*size + i];
    }
}

void binarize_weights_gpu(float *weights, int n, int size, float *binary)
{
    binarize_weights_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, size, binary);
    check_error(cudaPeekAtLastError());
}

__global__ void cuda_f32_to_f16(float* input_f32, size_t size, half *output_f16)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output_f16[idx] = __float2half(input_f32[idx]);
    //if (idx < size) *((unsigned short *)output_f16 + idx) = __float2half(input_f32[idx]);
}

void cuda_convert_f32_to_f16(float* input_f32, size_t size, float *output_f16) {
    cuda_f32_to_f16 <<< size / BLOCK + 1, BLOCK, 0, get_cuda_stream() >>> (input_f32, size, (half *)output_f16);
    check_error(cudaPeekAtLastError());
}

__global__ void cuda_f16_to_f32(half* input_f16, size_t size, float *output_f32)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output_f32[idx] = __half2float(input_f16[idx]);
    //if (idx < size) output_f32[idx] = __half2float(*((unsigned short *)input_f16 + idx));
}

void cuda_convert_f16_to_f32(float* input_f16, size_t size, float *output_f32) {
    cuda_f16_to_f32 <<< size / BLOCK + 1, BLOCK, 0, get_cuda_stream() >>> ((half *)input_f16, size, output_f32);
    check_error(cudaPeekAtLastError());
}

half *cuda_make_f16_from_f32_array(float *src, size_t n)
{
    half *dst16;
    size_t size = sizeof(half)*n;
    check_error(cudaMalloc((void **)&dst16, size));
    if (src) {
        cuda_convert_f32_to_f16(src, n, (float *)dst16);
    }
    if (!dst16) error("Cuda malloc failed\n");
    return dst16;
}

void forward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if(l.binary){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
    }

    if(l.xnor){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
        binarize_gpu(net.input_gpu, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
        net.input_gpu = l.binary_input_gpu;
    }

#ifdef CUDNN
    float one = 1;    // alpha[0], beta[0] is float for HALF and FLOAT
    float alpha = 1, beta = 0; 

#ifdef CUDNN_HALF
    // Note: For improved performance it is advised to use beta[0] = 0.0. 
    // For Tensor Core: cudnnSetConvolutionMathType() where cudnnMathType_t mathType = CUDNN_TENSOR_OP_MATH;
    // 1. or CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM and use CUDNN_DATA_HALF
    // 2. or CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
    // More: http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#tensor_ops

    const size_t input16_size = l.batch*l.c*l.w*l.h;
    const size_t output16_size = l.batch*l.out_c*l.out_h*l.out_w;

    if (*net.max_input16_size < input16_size) {
        //printf("\n input16_size: cur = %zu \t max = %zu \n", input16_size, *net.max_input16_size);
        *net.max_input16_size = input16_size;
        if (*net.input16_gpu) cuda_free(*net.input16_gpu);
        *net.input16_gpu = (float *)cuda_make_f16_from_f32_array(NULL, *net.max_input16_size);
    }
    float *input16 = *net.input16_gpu;

    if (*net.max_output16_size < output16_size) {
        *net.max_output16_size = output16_size;
        if (*net.output16_gpu) cuda_free(*net.output16_gpu);
        *net.output16_gpu = (float *)cuda_make_f16_from_f32_array(NULL, *net.max_output16_size);
    }
    float *output16 = *net.output16_gpu;

    cuda_convert_f32_to_f16(net.input_gpu, input16_size, input16);

    cudnnConvolutionForward(cudnn_handle(),
        &alpha,
        l.srcTensorDesc,
        input16,
        l.weightDesc,
        l.weights_gpu16,
        l.convDesc,
        l.fw_algo,
        net.workspace,
        l.workspace_size,
        &beta,
        l.dstTensorDesc,
        output16);

    if (l.batch_normalize){
        if (net.train){ // Training
            copy_gpu(l.outputs*l.batch / 2, output16, 1, l.x_gpu, 1);
            float zero = 0;
            // Batch-normalization can still take FP16 inputs and outputs, saving half the bandwidth
            // compared to FP32, it is just that the statistics and value adjustment should be done in FP32.
            cudnnBatchNormalizationForwardTraining(cudnn_handle(),
                CUDNN_BATCHNORM_SPATIAL,
                &one,
                &zero,
                l.normDstTensorDescF16,
                l.x_gpu,            // input
                l.normDstTensorDescF16,
                output16,            // output
                l.normTensorDesc,
                l.scales_gpu,
                l.biases_gpu,
                .01,
                l.rolling_mean_gpu,        // output (should be FP32)
                l.rolling_variance_gpu,    // output (should be FP32)
                .00001,
                l.mean_gpu,            // output (should be FP32)
                l.variance_gpu);    // output (should be FP32)

            cuda_convert_f16_to_f32(output16, output16_size, l.output_gpu);
            //forward_batchnorm_layer_gpu(l, net);
        }
        else{ // Detection
            cuda_convert_f16_to_f32(output16, output16_size, l.output_gpu);
            normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
            scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
            add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
        }
    }
    else{ // BIAS only
        cuda_convert_f16_to_f32(output16, output16_size, l.output_gpu);
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }

#else

    cudnnConvolutionForward(cudnn_handle(),
                &one,
                l.srcTensorDesc,
                net.input_gpu,
                l.weightDesc,
                l.weights_gpu,
                l.convDesc,
                l.fw_algo,
                net.workspace,
                l.workspace_size,
                &one,
                l.dstTensorDesc,
                l.output_gpu);
#endif    // CUDNN_HALF


#else
    int i, j;
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights_gpu + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output_gpu + (i*l.groups + j)*n*m;
            float *im = net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1){
                b = im;
            } else {
                im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
#endif

#ifndef CUDNN_HALF
    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
#endif // no CUDNN_HALF

    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
    //if(l.dot > 0) dot_error_gpu(l);
    if(l.binary || l.xnor) swap_binary(&l);
}

__global__ void smooth_kernel(float *x, int n, int w, int h, int c, int size, float rate, float *delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -(size/2.f);
    int h_offset = -(size/2.f);

    int out_index = j + w*(i + h*(k + c*b));
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i + l;
            int cur_w = w_offset + j + m;
            int index = cur_w + w*(cur_h + h*(k + b*c));
            int valid = (cur_h >= 0 && cur_h < h &&
                    cur_w >= 0 && cur_w < w);
            delta[out_index] += valid ? rate*(x[index] - x[out_index]) : 0;
        }
    }
}

extern "C" void smooth_layer(layer l, int size, float rate)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;

    size_t n = h*w*c*l.batch;

    smooth_kernel<<<cuda_gridsize(n), BLOCK>>>(l.output_gpu, n, l.w, l.h, l.c, size, rate, l.delta_gpu);
    check_error(cudaPeekAtLastError());
}

void backward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    if(l.smooth){
        smooth_layer(l, 5, l.smooth);
    }
    //constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);


#ifndef CUDNN_HALF
    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
#endif // no CUDNN_HALF
    float *original_input = net.input_gpu;

    if(l.xnor) net.input_gpu = l.binary_input_gpu;
#ifdef CUDNN
    float one = 1;
    float alpha = 1, beta = 0;

#ifdef CUDNN_HALF
    const size_t input16_size = l.batch*l.c*l.w*l.h;
    const size_t delta16_size = l.batch*l.n*l.out_w*l.out_h;
    
    if (*net.max_input16_size < input16_size) {
        *net.max_input16_size = input16_size;
        if(*net.input16_gpu) cuda_free(*net.input16_gpu);
        *net.input16_gpu = (float *)cuda_make_f16_from_f32_array(NULL, *net.max_input16_size);
    }
    float *input16 = *net.input16_gpu;

    if (*net.max_output16_size < delta16_size) {
        *net.max_output16_size = delta16_size;
        if(*net.output16_gpu) cuda_free(*net.output16_gpu);
        *net.output16_gpu = (float *)cuda_make_f16_from_f32_array(NULL, *net.max_output16_size);
    }
    float *delta16 = *net.output16_gpu;

    cuda_convert_f32_to_f16(net.input_gpu, input16_size, input16);
    cuda_convert_f32_to_f16(l.delta_gpu, delta16_size, delta16);

    if (l.batch_normalize) {
        float one = 1;
        float zero = 0;
        cudnnBatchNormalizationBackward(cudnn_handle(),
            CUDNN_BATCHNORM_SPATIAL,
            &one,
            &zero,
            &one,
            &one,
            l.normDstTensorDescF16,
            l.x_gpu,                // input
            l.normDstTensorDescF16,
            delta16,                // input
            l.normDstTensorDescF16,
            l.x_norm_gpu,            // output
            l.normTensorDesc,
            l.scales_gpu,            // output (should be FP32)
            l.scale_updates_gpu,    // output (should be FP32)
            l.bias_updates_gpu,        // output (should be FP32)
            .00001,
            l.mean_gpu,                // input (should be FP32)
            l.variance_gpu);        // input (should be FP32)
        copy_gpu(l.outputs*l.batch / 2, l.x_norm_gpu, 1, delta16, 1);
    }
    else{
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }

    // convert input: net.input (x), l.delta_gpu (y) from fp32 to fp16
    // get output: l.weight_updates_gpu (dw) and convert it to fp32 (ONLY if it is fp16)

    // calculate conv weight updates
    // Already: l.weight_updates_gpu = (l.weight_updates_gpu - l.weight*decay*batch*subdivision)*momentum
    //   so we should copy f32 to f16, or compute: f16=(w_up - w*d*b*s)*m
    cuda_convert_f32_to_f16(l.weight_updates_gpu, l.c*l.n*l.size*l.size, l.weight_updates_gpu16);

    cudnnConvolutionBackwardFilter(cudnn_handle(),
        &one,
        l.srcTensorDesc,
        input16, //net.input,
        l.ddstTensorDesc,
        delta16, //l.delta_gpu,
        l.convDesc,
        l.bf_algo,
        net.workspace,
        l.workspace_size,
        &one,
        l.dweightDesc,
        l.weight_updates_gpu16);    // l.weight_updates_gpu);

    cuda_convert_f16_to_f32(l.weight_updates_gpu16, l.c*l.n*l.size*l.size, l.weight_updates_gpu);

    if (net.delta_gpu) {
        if (l.binary || l.xnor) swap_binary(&l);

        // http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardData
        // calculate delta for the next layer
        // convert input: l.weights_gpu (w), l.delta_gpu (dy) from fp32 to fp16
        // get output: net.delta (dx) and convert it to fp32 (ONLY if it is fp16)
        cudnnConvolutionBackwardData(cudnn_handle(),
            &alpha,
            l.weightDesc,
            l.weights_gpu16, //l.weights_gpu,
            l.ddstTensorDesc,
            delta16, //l.delta_gpu,
            l.convDesc,
            l.bd_algo,
            net.workspace,
            l.workspace_size,
            &beta,
            l.dsrcTensorDesc,
            input16);    // net.delta_gpu);

        cuda_convert_f16_to_f32(input16, input16_size, net.delta_gpu);

        if (l.binary || l.xnor) swap_binary(&l);
        if (l.xnor) gradient_array_gpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, net.delta_gpu);
    }
#else    // CUDNN_HALF

    // calculate conv weight updates
    // if used: beta=1 then loss decreases faster
    cudnnConvolutionBackwardFilter(cudnn_handle(),
            &one,
            l.srcTensorDesc,
            net.input_gpu,
            l.ddstTensorDesc,
            l.delta_gpu,
            l.convDesc,
            l.bf_algo,
            net.workspace,
            l.workspace_size,
            &one,
            l.dweightDesc,
            l.weight_updates_gpu);

    if(net.delta_gpu){
        if(l.binary || l.xnor) swap_binary(&l);
        // http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardData
        // calculate delta for the next layer
        cudnnConvolutionBackwardData(cudnn_handle(),
                &one,
                l.weightDesc,
                l.weights_gpu,
                l.ddstTensorDesc,
                l.delta_gpu,
                l.convDesc,
                l.bd_algo,
                net.workspace,
                l.workspace_size,
                &one,
                l.dsrcTensorDesc,
                net.delta_gpu);
        if(l.binary || l.xnor) swap_binary(&l);
        if(l.xnor) gradient_array_gpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, net.delta_gpu);
    }

#endif    // CUDNN_HALF

#else    // CUDNN
    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    int i, j;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.delta_gpu + (i*l.groups + j)*m*k;
            float *b = net.workspace;
            float *c = l.weight_updates_gpu + j*l.nweights/l.groups;

            float *im  = net.input_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;
            float *imd = net.delta_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;

            im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if (net.delta_gpu) {
                if (l.binary || l.xnor) swap_binary(&l);
                a = l.weights_gpu + j*l.nweights/l.groups;
                b = l.delta_gpu + (i*l.groups + j)*m*k;
                c = net.workspace;
                if (l.size == 1) {
                    c = imd;
                }

                gemm_gpu(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (l.size != 1) {
                    col2im_gpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
                }
                if(l.binary || l.xnor) {
                    swap_binary(&l);
                }
            }
            if(l.xnor) gradient_array_gpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, net.delta_gpu + i*l.c*l.h*l.w);
        }
    }
#endif
}

void pull_convolutional_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    cuda_pull_array(l.biases_gpu, l.biases, l.n);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.n);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void push_convolutional_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
#ifdef CUDNN_HALF
    cuda_convert_f32_to_f16(l.weights_gpu, l.nweights, l.weights_gpu16);
#endif
    cuda_push_array(l.biases_gpu, l.biases, l.n);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.n);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void update_convolutional_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        }
    }else{
        // update weights:
        // weights_gpu = weights_gpu*(1 - decay*lr) + weight_updates_gpu*lr / (batch*subdivision) =
        //  weights_gpu*(1 - 0.0005*0.001) + weight_updates_gpu*0.001/(64*8) = 
        //  weights_gpu * 0.999 999 5 + weight_updates_gpu * 0.000 001 953125
        // 
        // weight_updates_gpu = (weight_updates_gpu - weights_gpu*decay*batch*subdivision)*momentum = 
        //  (weight_updates_gpu - weights_gpu * 0.0005 * 64 * 8) * 0.9 = 
        //  weight_updates_gpu*0.9 - weights_gpu*0.2304
        axpy_gpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.nweights, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1);

        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);

        if(l.scales_gpu){
            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
        }
    }
    if(l.clip){
        constrain_gpu(l.nweights, l.clip, l.weights_gpu, 1);
    }
}


