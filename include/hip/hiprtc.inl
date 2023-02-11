#ifndef NV_HIP_RTC_H
#define NV_HIP_RTC_H

#include <nvrtc.h>

#if _WIN32
#define NV_HIPRTC_EXPORT __declspec(dllexport)
#else
#define NV_HIPRTC_EXPORT __attribute__((visibility("default")))
#endif

#ifdef NV_HIPRTC_RUNTIME_LIB_MODE
#define NV_HIPRTC_DECORATOR NV_HIPRTC_EXPORT
#else
#define NV_HIPRTC_DECORATOR
#endif

#if __cplusplus
extern "C" {
#endif

typedef nvrtcProgram hiprtcProgram;

typedef enum {
  HIPRTC_SUCCESS = 0,
  HIPRTC_ERROR_OUT_OF_MEMORY,
  HIPRTC_ERROR_PROGRAM_CREATION_FAILURE,
  HIPRTC_ERROR_INVALID_INPUT
} hiprtcResult;

NV_HIPRTC_DECORATOR inline hiprtcResult
nvrtcError2hiprtcError(nvrtcResult error) {
  switch (error) {
  case NVRTC_SUCCESS:
    return HIPRTC_SUCCESS;
  default:
    return HIPRTC_ERROR_OUT_OF_MEMORY;
  }
}

NV_HIPRTC_DECORATOR
const char *hiprtcGetErrorString(hiprtcResult result)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  switch (result) {
  case HIPRTC_SUCCESS:
    return "HIPRTC_SUCCESS";
  default:
    return "HIPRTC_ERROR_OUT_OF_MEMORY";
  }
}
#endif

NV_HIPRTC_DECORATOR
hiprtcResult hiprtcCreateProgram(hiprtcProgram *prog, const char *src,
                                 const char *name, int numHeaders,
                                 const char *const *headers,
                                 const char *const *includeNames)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return nvrtcError2hiprtcError(
      nvrtcCreateProgram(prog, src, name, numHeaders, headers, includeNames));
}
#endif

NV_HIPRTC_DECORATOR
hiprtcResult hiprtcDestroyProgram(hiprtcProgram *prog)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return nvrtcError2hiprtcError(nvrtcDestroyProgram(prog));
}
#endif

NV_HIPRTC_DECORATOR
hiprtcResult hiprtcAddNameExpression(hiprtcProgram prog,
                                     const char *const name_expression)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return nvrtcError2hiprtcError(nvrtcAddNameExpression(prog, name_expression));
}
#endif

NV_HIPRTC_DECORATOR
hiprtcResult hiprtcCompileProgram(hiprtcProgram prog, int numOptions,
                                  const char *const *options)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return nvrtcError2hiprtcError(nvrtcCompileProgram(prog, numOptions, options));
}
#endif

NV_HIPRTC_DECORATOR
hiprtcResult hiprtcGetProgramLog(hiprtcProgram prog, char *log)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return nvrtcError2hiprtcError(nvrtcGetProgramLog(prog, log));
}
#endif

NV_HIPRTC_DECORATOR
hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram prog, size_t *logSize)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return nvrtcError2hiprtcError(nvrtcGetProgramLogSize(prog, logSize));
}
#endif

NV_HIPRTC_DECORATOR
hiprtcResult hiprtcGetCode(hiprtcProgram prog, char *ptx)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return nvrtcError2hiprtcError(nvrtcGetPTX(prog, ptx));
}
#endif

NV_HIPRTC_DECORATOR
hiprtcResult hiprtcGetCodeSize(hiprtcProgram prog, size_t *ptxSizeRet)
#ifdef NV_HIP_RUNTIME_LIB_MODE
    ;
#else
{
  return nvrtcError2hiprtcError(nvrtcGetPTXSize(prog, ptxSizeRet));
}
#endif

#if __cplusplus
}
#endif
#endif