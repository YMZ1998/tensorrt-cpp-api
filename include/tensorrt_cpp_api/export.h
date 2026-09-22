#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)

#if defined(TRT_CPP_API_BUILD_SHARED)
#if defined(tensorrt_cpp_api_EXPORTS)
#define TRT_CPP_API_EXPORT __declspec(dllexport)
#else
#define TRT_CPP_API_EXPORT __declspec(dllimport)
#endif
#else
#define TRT_CPP_API_EXPORT
#endif

#else

#define TRT_CPP_API_EXPORT

#endif
