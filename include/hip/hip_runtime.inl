#ifndef NV_HIP_RUNTIME_H
#define NV_HIP_RUNTIME_H

#include <cuda.h>
#include <cuda_runtime_api.h>

#if __cplusplus
extern "C" {
#endif

#if _WIN32
#define NV_HIP_EXPORT __declspec(dllexport)
#else
#define NV_HIP_EXPORT __attribute__((visibility("default")))
#endif

#define HOST_VISIBLE __host__
#define DEVICE_VISIBLE __device__
#define HOST_DEVICE_VISIBLE __host__ __device__

#ifdef NV_HIP_RUNTIME_LIB_MODE
#define NV_HIP_DECORATOR NV_HIP_EXPORT HOST_VISIBLE
#define NV_HIP_DECORATOR_D NV_HIP_DECORATOR
#define NV_HIP_DECORATOR_HD NV_HIP_DECORATOR
#else
#define NV_HIP_DECORATOR HOST_VISIBLE
#define NV_HIP_DECORATOR_D DEVICE_VISIBLE
#define NV_HIP_DECORATOR_HD HOST_DEVICE_VISIBLE
#endif

enum hipError_t { hipSuccess, hipErrorInvalidValue };

enum hipMemcpyKind {
  hipMemcpyHostToHost = 0,
  hipMemcpyHostToDevice = 1,
  hipMemcpyDeviceToHost = 2,
  hipMemcpyDeviceToDevice = 3,
  hipMemcpyDefault = 4
};

enum hipDeviceAttribute_t {
  hipDeviceAttributeComputeCapabilityMajor = 23,
  hipDeviceAttributeComputeCapabilityMinor = 61
};

enum hipLimit_t {
  hipLimitStackSize = 0x0,
  hipLimitPrintfFifoSize = 0x01,
  hipLimitMallocHeapSize = 0x02,
  hipLimitRange
};

typedef cudaStream_t hipStream_t;
typedef cudaEvent_t hipEvent_t;
typedef cudaDeviceProp hipDeviceProp_t;
typedef cudaPointerAttributes hipPointerAttributes_t;
typedef CUmodule hipModule_t;
typedef CUfunction hipFunction_t;
typedef CUmodule hipModule_t;

NV_HIP_DECORATOR_HD inline hipError_t cudaError2hipError(cudaError_t error) {
  switch (error) {
  case cudaSuccess:
    return hipSuccess;
  default:
    return hipErrorInvalidValue;
  }
}

NV_HIP_DECORATOR_HD inline cudaError_t hipError2cudaError(hipError_t error) {
  switch (error) {
  case hipSuccess:
    return cudaSuccess;
  default:
    return cudaErrorInvalidValue;
  }
}

NV_HIP_DECORATOR_HD inline cudaDeviceAttr
hipDevAttr2cudaDevAttr(hipDeviceAttribute_t attr) {
  switch (attr) {
  case hipDeviceAttributeComputeCapabilityMajor:
    return cudaDevAttrComputeCapabilityMajor;
  case hipDeviceAttributeComputeCapabilityMinor:
    return cudaDevAttrComputeCapabilityMinor;
  default:
    return cudaDevAttrComputeCapabilityMinor;
  }
}

NV_HIP_DECORATOR_HD inline cudaLimit hipLimit2cudaLimit(hipLimit_t limit) {
  switch (limit) {
  case hipLimitStackSize:
    return cudaLimitStackSize;
  case hipLimitPrintfFifoSize:
    return cudaLimitPrintfFifoSize;
  case hipLimitMallocHeapSize:
    return cudaLimitMallocHeapSize;
  default:
    return cudaLimitMallocHeapSize;
  }
}

NV_HIP_DECORATOR_HD inline hipError_t cuError2hipError(CUresult error) {
  switch (error) {
  case CUDA_SUCCESS:
    return hipSuccess;
  default:
    return hipErrorInvalidValue;
  }
}

NV_HIP_DECORATOR_HD
inline cudaMemcpyKind hipMemcpyKind2cudaMemcpyKind(hipMemcpyKind direction) {
  switch (direction) {
  case hipMemcpyDefault:
    return cudaMemcpyDefault;
  case hipMemcpyDeviceToHost:
    return cudaMemcpyDeviceToHost;
  case hipMemcpyHostToDevice:
    return cudaMemcpyHostToDevice;
  case hipMemcpyHostToHost:
    return cudaMemcpyHostToHost;
  default:
    return cudaMemcpyDeviceToDevice;
  }
}

NV_HIP_DECORATOR_HD
hipError_t hipMalloc(void **ptr, size_t size)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaMalloc(ptr, size));
}
#endif

NV_HIP_DECORATOR_HD
hipError_t hipFree(void *ptr)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaFree(ptr));
}
#endif

NV_HIP_DECORATOR_HD
hipError_t hipMallocPitch(void **ptr, size_t *pitch, size_t width,
                          size_t height)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaMallocPitch(ptr, pitch, width, height));
}
#endif

NV_HIP_DECORATOR
hipError_t hipMemcpy(void *dst, void *src, size_t size, hipMemcpyKind direction)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(
      cudaMemcpy(dst, src, size, hipMemcpyKind2cudaMemcpyKind(direction)));
}
#endif

NV_HIP_DECORATOR
hipError_t hipStreamCreate(hipStream_t *stream)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaStreamCreate(stream));
}
#endif

NV_HIP_DECORATOR
hipError_t hipStreamDestroy(hipStream_t stream)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaStreamDestroy(stream));
}
#endif

NV_HIP_DECORATOR
hipError_t hipStreamSynchronize(hipStream_t stream)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaStreamSynchronize(stream));
}
#endif

NV_HIP_DECORATOR
hipError_t hipMemcpyAsync(void *dst, void *src, size_t size,
                          hipMemcpyKind direction, hipStream_t stream)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaMemcpyAsync(
      dst, src, size, hipMemcpyKind2cudaMemcpyKind(direction), stream));
}
#endif

NV_HIP_DECORATOR
hipError_t hipMemGetInfo(size_t *free, size_t *total)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaMemGetInfo(free, total));
}
#endif

// TODO implement this
NV_HIP_DECORATOR
const char *hipGetErrorString(hipError_t error)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaGetErrorString(cudaSuccess);
}
#endif

NV_HIP_DECORATOR
hipError_t hipEventCreateWithFlags(hipEvent_t *event, unsigned int flags)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaEventCreateWithFlags(event, flags));
}
#endif

NV_HIP_DECORATOR
hipError_t hipEventCreate(hipEvent_t *event)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaEventCreate(event));
}
#endif

NV_HIP_DECORATOR
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaEventRecord(event, stream));
}
#endif

NV_HIP_DECORATOR
hipError_t hipEventDestroy(hipEvent_t event)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaEventDestroy(event));
}
#endif

NV_HIP_DECORATOR
hipError_t hipEventSynchronize(hipEvent_t event)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaEventSynchronize(event));
}
#endif

NV_HIP_DECORATOR
hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t end)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaEventElapsedTime(ms, start, end));
}
#endif

NV_HIP_DECORATOR
hipError_t hipSetDevice(int device)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaSetDevice(device));
}
#endif

NV_HIP_DECORATOR
hipError_t hipGetDevice(int *device)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaGetDevice(device));
}
#endif

NV_HIP_DECORATOR
hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int device)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaGetDeviceProperties(prop, device));
}
#endif

NV_HIP_DECORATOR
hipError_t hipPointerGetAttributes(hipPointerAttributes_t *attributes,
                                   const void *ptr)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaPointerGetAttributes(attributes, ptr));
}
#endif

NV_HIP_DECORATOR
hipError_t hipModuleLoadData(hipModule_t *module, const void *image)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cuError2hipError(cuModuleLoadData(module, image));
}
#endif

NV_HIP_DECORATOR
hipError_t hipModuleGetFunction(hipFunction_t *hfunc, hipModule_t hmod,
                                const char *name)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cuError2hipError(cuModuleGetFunction(hfunc, hmod, name));
}
#endif

NV_HIP_DECORATOR
hipError_t hipModuleUnload(hipModule_t hmod)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cuError2hipError(cuModuleUnload(hmod));
}
#endif

NV_HIP_DECORATOR
hipError_t
hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX,
                      unsigned int gridDimY, unsigned int gridDimZ,
                      unsigned int blockDimX, unsigned int blockDimY,
                      unsigned int blockDimZ, unsigned int sharedMemBytes,
                      hipStream_t hStream, void **kernelParams, void **extra)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cuError2hipError(
      cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                     blockDimZ, sharedMemBytes, hStream, kernelParams, extra));
}
#endif

NV_HIP_DECORATOR
hipError_t hipDeviceSynchronize()
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaDeviceSynchronize());
}
#endif

NV_HIP_DECORATOR
hipError_t hipInit(unsigned int flags)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cuError2hipError(cuInit(flags));
}
#endif

NV_HIP_DECORATOR
hipError_t hipGetDeviceCount(int *count)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaGetDeviceCount(count));
}
#endif

NV_HIP_DECORATOR
hipError_t hipDeviceGetAttribute(int *val, hipDeviceAttribute_t attr,
                                 int device)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(
      cudaDeviceGetAttribute(val, hipDevAttr2cudaDevAttr(attr), device));
}
#endif

NV_HIP_DECORATOR
hipError_t hipDeviceSetLimit(hipLimit_t limit, size_t value)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(
      cudaDeviceSetLimit(hipLimit2cudaLimit(limit), value));
}
#endif

NV_HIP_DECORATOR
hipError_t hipDriverGetVersion(int *version)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaDriverGetVersion(version));
}
#endif

NV_HIP_DECORATOR
hipError_t hipRuntimeGetVersion(int *version)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaError2hipError(cudaRuntimeGetVersion(version));
}
#endif

NV_HIP_DECORATOR
const char *hipGetErrorName(hipError_t error)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return cudaGetErrorName(hipError2cudaError(error));
}
#endif

#if __cplusplus
}
#endif

#endif