#include <hip/hip_runtime.h>
#include <iostream>

#define HIP_CHECK(call)                             \
  {                                                 \
    auto _res = (call);                             \
    if (_res != hipSuccess)                         \
    {                                               \
      std::cout << #call << " failed" << std::endl; \
    }                                               \
  }

__global__ void kernel(int *ptr, int set) { *ptr = set; }

int main()
{
  int *ptr{nullptr};
  constexpr int set{10};
  int res{0};
  HIP_CHECK(hipMalloc((void **)&ptr, sizeof(int)));
  kernel<<<1, 1>>>(ptr, set);
  HIP_CHECK(hipMemcpy((void *)&res, (void *)ptr, sizeof(int), hipMemcpyDeviceToHost));
  std::cout << "Res is :: " << ((res == set) ? "Correct" : "incorrect")
            << std::endl;
}
