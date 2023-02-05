#ifndef NV_HIP_RUNTIME_H
#define NV_HIP_RUNTIME_H

#include <cuda_runtime.h>

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
  hipMemcpyDefault,
  hipMemcpyDeviceToHost,
  hipMemcpyHostToDevice,
  hipMemcpyDeviceToDevice
};

NV_HIP_DECORATOR_HD
inline hipError_t cudaError2hipError(cudaError_t error) {
  switch (error) {
  case cudaSuccess:
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

#if __cplusplus
}
#endif

#endif