#include <faiss/gpu/GpuIndexScalarQuantizer.h>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>

namespace faiss {
namespace gpu {

GpuIndexScalarQuantizer::GpuIndexScalarQuantizer(
        GpuResourcesProvider* provider,
        const faiss::IndexScalarQuantizer* index,
        GpuIndexScalarQuantizerConfig config)
        : GpuIndex(
                provider->getResources(),
                index->d,
                index->metric_type,
                index->metric_arg, config),
        gpuCodes(provider->getResources().get(),
                makeDevAlloc(AllocType::Quantizer, 0),
                {(idx_t)index->codes.size()}) {
    GpuResources* res = provider->getResources().get();
    GpuScalarQuantizer* gpuSq = new GpuScalarQuantizer(res, index->sq);

    HostTensor<uint8_t, 1, true> cpuCodes(
        (uint8_t*)index->codes.data(), {(idx_t)index->codes.size()}
    );

    auto stream = res->getDefaultStreamCurrentDevice();
    gpuCodes.copyFrom(cpuCodes, stream);
}

GpuIndexScalarQuantizer::~GpuIndexScalarQuantizer() {}

} // namespace gpu
} // namespace faiss
