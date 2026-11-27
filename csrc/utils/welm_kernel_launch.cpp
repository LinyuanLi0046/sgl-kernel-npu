// Copyright (c) 2026. Licensed under the BSD 3-Clause License.
#include "welm_kernel_launch.h"

#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace sglang::npu_kernel::welm {
namespace {
constexpr const char *KERNEL_NAMES[] = {"welm_pack_decode_inputs_kernel", "welm_sigmoid_mul_kernel"};
struct Entry {
    aclrtBinHandle binary = nullptr;
    aclrtFuncHandle function = nullptr;
};
using Entries = std::array<Entry, 2>;

std::string BinaryPath(const char *kernelName)
{
    Dl_info info{};
    TORCH_CHECK(dladdr(reinterpret_cast<const void *>(&LaunchByName), &info) != 0 && info.dli_fname,
                "WeLM: cannot locate libsgl_kernel_npu.so");
    std::unique_ptr<char, decltype(&std::free)> library(realpath(info.dli_fname, nullptr), &std::free);
    TORCH_CHECK(library, "WeLM: cannot resolve library path: ", info.dli_fname);
    const std::string path(library.get());
    return path.substr(0, path.find_last_of('/') + 1) + kernelName + ".o";
}

aclrtFuncHandle GetFunction(Kernel kernel)
{
    aclrtContext context = nullptr;
    const auto contextStatus = aclrtGetCurrentContext(&context);
    TORCH_CHECK(contextStatus == ACL_SUCCESS && context, "WeLM: no current NPU context, ACL error ", contextStatus);
    const size_t index = static_cast<size_t>(kernel);
    // Same-context warm launches need neither a lock nor a filename lookup.
    thread_local aclrtContext lastContext = nullptr;
    thread_local std::array<aclrtFuncHandle, 2> lastFunctions{};
    if (lastContext != context) {
        lastContext = context;
        lastFunctions = {};
    }
    if (lastFunctions[index]) {
        return lastFunctions[index];
    }

    // Each runtime context owns its binary/function handles. Keep them alive
    // for captured graphs, and let runtime teardown reclaim them; calling ACL
    // in a static destructor can run after torch_npu has finalized the runtime.
    // This cache assumes contexts live for the worker process's lifetime;
    // restart the worker after resetting its NPU device.
    static std::mutex mutex;
    static std::unordered_map<aclrtContext, Entries> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto &entry = cache[context][index];
    if (!entry.function) {
        const char *name = KERNEL_NAMES[index];
        const auto path = BinaryPath(name);
        // CANN rejects MAGIC options for LoadFromFile (supported by the data
        // loading API instead). File loading uses the ELF metadata; the build
        // checker requires K_TYPE_AIV on our ordinary entry.
        auto status = aclrtBinaryLoadFromFile(path.c_str(), nullptr, &entry.binary);
        TORCH_CHECK(status == ACL_SUCCESS, "WeLM: loading ", path, " failed, ACL error ", status);
        // Several SK functions carry the same tiling key. Resolve ONLY the
        // ordinary global by name; never use tilingKey=0 to select a function.
        status = aclrtBinaryGetFunction(entry.binary, name, &entry.function);
        if (status != ACL_SUCCESS) {
            aclrtBinaryUnLoad(entry.binary);
            entry = {};
            TORCH_CHECK(false, "WeLM: resolving ", name, " in ", path, " failed, ACL error ", status);
        }
    }
    lastFunctions[index] = entry.function;
    return entry.function;
}
}  // namespace

aclError LaunchByName(Kernel kernel, uint32_t blockDim, aclrtStream stream, void *args, size_t argsSize)
{
    return aclrtLaunchKernelWithHostArgs(GetFunction(kernel), blockDim, stream, nullptr,
                                      args, argsSize, nullptr, 0);
}
}  // namespace sglang::npu_kernel::welm
