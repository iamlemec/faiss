#include <faiss/gpu/GpuIndexScalarQuantizer.h>
#include <faiss/gpu/utils/HostTensor.cuh>

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
    gpuSq = new GpuScalarQuantizer(res, index->sq);

    HostTensor<uint8_t, 1, true> cpuCodes(
        (uint8_t*)index->codes.data(), {(idx_t)index->codes.size()}
    );

    auto stream = res->getDefaultStreamCurrentDevice();
    gpuCodes.copyFrom(cpuCodes, stream);
}

GpuIndexScalarQuantizer::~GpuIndexScalarQuantizer() {}

void GpuIndexScalarQuantizer::reconstruct_batch(idx_t n, const idx_t* keys, float* recons) const {
    Codec<ScalarQuantizer::QuantizerType::QT_8bit, 1> codec(
        gpuSq->code_size,
        gpuSq->gpuTrained.data(),
        gpuSq->gpuTrained.data() + gpuSq->d);
}

void GpuIndexScalarQuantizer::train(idx_t n, const float* x) {}

void GpuIndexScalarQuantizer::reset() {}

bool GpuIndexScalarQuantizer::addImplRequiresIDs_() const {
    return false;
};

void GpuIndexScalarQuantizer::addImpl_(idx_t n, const float* x, const idx_t* ids) {};

void GpuIndexScalarQuantizer::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {};

} // namespace gpu
} // namespace faiss
