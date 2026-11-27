// Copyright (c) 2026. Licensed under the BSD 3-Clause License.
#pragma once

#include "acl/acl.h"
#include <c10/util/Exception.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace sglang::npu_kernel::welm {
enum class Kernel { Pack, SigmoidMul };

aclError LaunchByName(Kernel kernel, uint32_t blockDim, aclrtStream stream, void *args, size_t argsSize);

template <typename Args>
void Launch(Kernel kernel, const char *opName, uint32_t blockDim, Args args)
{
    // Match EXEC_KERNEL_CMD's queue/stream behavior. The callback may run after
    // this function returns: own the scalar/pointer payload, never its address.
    // Resolve the raw stream before enqueueing (stream(true) would drain the queue).
    const auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto call = [kernel, blockDim, stream, args]() mutable -> int {
        return LaunchByName(kernel, blockDim, stream, &args, sizeof(args));
    };
    at_npu::native::OpCommand::RunOpApi(opName, call);
}
}  // namespace sglang::npu_kernel::welm
