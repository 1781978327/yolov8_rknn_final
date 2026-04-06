#include <unistd.h>
#include <iostream>
#include <string.h>
#include <queue>
#include "rknn_fp.h"

rknn_fp::rknn_fp(const char *model_path, int cpuid, rknn_core_mask core_mask,
                 int n_input, int n_output)
{
    int ret = 0;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpuid, &mask);
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
        std::cerr << "set thread affinity failed" << std::endl;
    printf("Bind NPU process on CPU %d\n", cpuid);
    _cpu_id   = cpuid;
    _n_input  = n_input;
    _n_output = n_output;

    FILE *fp = fopen(model_path, "rb");
    if(fp == NULL) {
        printf("fopen %s fail!\n", model_path);
        exit(-1);
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    void *model = malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", model_path);
        free(model);
        exit(-1);
    }
    fclose(fp);

    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    if(ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        exit(-1);
    }
    ret = rknn_set_core_mask(ctx, core_mask);
    if(ret < 0) {
        printf("set NPU core_mask fail! ret=%d\n", ret);
        exit(-1);
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    printf("api version: %s\n", version.api_version);
    printf("driver version: %s\n", version.drv_version);

    memset(_input_attrs, 0, _n_input * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < _n_input; i++) {
        _input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(_input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_init error! ret=%d\n", ret);
            exit(-1);
        }
        dump_tensor_attr(&_input_attrs[i]);
    }

    rknn_tensor_type   input_type   = RKNN_TENSOR_UINT8;
    rknn_tensor_format input_layout = RKNN_TENSOR_NHWC;
    _input_attrs[0].type = input_type;
    _input_attrs[0].fmt = input_layout;
    _input_mems[0] = rknn_create_mem(ctx, _input_attrs[0].size_with_stride);

    memset(_output_attrs, 0, _n_output * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < _n_output; i++) {
        _output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(_output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            exit(-1);
        }
        dump_tensor_attr(&_output_attrs[i]);
    }

    for (uint32_t i = 0; i < _n_output; ++i) {
        int output_size = _output_attrs[i].n_elems * sizeof(float);
        _output_mems[i]  = rknn_create_mem(ctx, output_size);
    }

    ret = rknn_set_io_mem(ctx, _input_mems[0], &_input_attrs[0]);
    if (ret < 0) {
        printf("rknn_set_io_mem fail! ret=%d\n", ret);
        exit(-1);
    }

    for (uint32_t i = 0; i < _n_output; ++i) {
        _output_attrs[i].type = RKNN_TENSOR_FLOAT32;
        ret = rknn_set_io_mem(ctx, _output_mems[i], &_output_attrs[i]);
        if (ret < 0) {
            printf("rknn_set_io_mem fail! ret=%d\n", ret);
            exit(-1);
        }
    }
}

rknn_fp::~rknn_fp(){
    rknn_destroy(ctx);
}

void rknn_fp::dump_tensor_attr(rknn_tensor_attr* attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
            "zp=%d, scale=%f\n",
            attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
            attr->n_elems, attr->size,
            attr->fmt == RKNN_TENSOR_NHWC ? "NHWC" : "NCHW",
            attr->type == RKNN_TENSOR_FLOAT32 ? "FP32" :
            attr->type == RKNN_TENSOR_UINT8  ? "UINT8" : "OTHER",
            attr->qnt_type == RKNN_TENSOR_QNT_NONE ? "NONE" :
            attr->qnt_type == RKNN_TENSOR_QNT_DFP ? "DFP" :
            attr->qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC ? "AFFINE" : "UNKNOWN",
            attr->zp, attr->scale);
}

int rknn_fp::inference(unsigned char *data){
    int ret;
    int width = _input_attrs[0].dims[2];
    memcpy(_input_mems[0]->virt_addr, data, width * _input_attrs[0].dims[1] * _input_attrs[0].dims[3]);

    ret = rknn_run(ctx, nullptr);
    if(ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }
    rknn_perf_run perf_run;
    ret = rknn_query(ctx, RKNN_QUERY_PERF_RUN, &perf_run, sizeof(perf_run));

    for(int i=0;i<_n_output;i++){
        _output_buff[i] = (float*)_output_mems[i]->virt_addr;
    }
    return perf_run.run_duration;
}

float rknn_fp::cal_NPU_performance(std::queue<float> &history_time, float &sum_time, float cost_time){
    if(history_time.size()<10){
        history_time.push(cost_time);
        sum_time += cost_time;
    }
    else if(history_time.size()==10){
        sum_time -= history_time.front();
        sum_time += cost_time;
        history_time.pop();
        history_time.push(cost_time);
    }
    else{
        printf("cal_NPU_performance Error\n");
        return -1;
    }
    return sum_time / history_time.size();
}
