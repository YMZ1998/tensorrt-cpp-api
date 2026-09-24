#pragma once

#include <filesystem>

#define NOMINMAX
#include <cuda_runtime_api.h>
#include <windows.h>

namespace trtcpp {

std::string getCpuName() {
    HKEY hKey = nullptr;
    char cpuName[256] = {};
    DWORD size = sizeof(cpuName);

    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &hKey) != ERROR_SUCCESS) {
        return "Unknown CPU";
    }

    RegQueryValueExA(hKey, "ProcessorNameString", nullptr, nullptr, reinterpret_cast<LPBYTE>(cpuName), &size);

    RegCloseKey(hKey);

    return cpuName;
}

std::string getGpuName() {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
        return "No CUDA GPU";
    }

    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return "Unknown GPU";
    }

    return prop.name;
}

std::filesystem::path getExeDir() {
    wchar_t buffer[MAX_PATH];
    GetModuleFileNameW(nullptr, buffer, MAX_PATH);

    return std::filesystem::path(buffer).parent_path();
}

} // namespace common
