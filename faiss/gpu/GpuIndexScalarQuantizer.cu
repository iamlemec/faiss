#include <faiss/gpu/GpuIndexScalarQuantizer.h>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
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
        sq(index->sq),
        gpuCodes(provider->getResources().get(),
                makeDevAlloc(AllocType::Quantizer, 0),
                {(idx_t)index->codes.size()}),
        gpuTrained(provider->getResources().get(),
                makeDevAlloc(AllocType::Quantizer, 0),
                {(idx_t)index->sq.trained.size()}) {

    GpuResources* res = provider->getResources().get();
    auto stream = res->getDefaultStreamCurrentDevice();

    HostTensor<float, 1, true> cpuTrained(
        (float*)sq.trained.data(), {(idx_t)sq.trained.size()}
    );
    gpuTrained.copyFrom(cpuTrained, stream);

    HostTensor<uint8_t, 1, true> cpuCodes(
        (uint8_t*)index->codes.data(), {(idx_t)index->codes.size()}
    );
    gpuCodes.copyFrom(cpuCodes, stream);
}

GpuIndexScalarQuantizer::~GpuIndexScalarQuantizer() {}

void GpuIndexScalarQuantizer::reconstruct_batch(idx_t n, const idx_t* keys, float* out) const {
    // get gpu resources
    DeviceScope scope(config_.device);
    auto stream = resources_->getDefaultStream(config_.device);

    // number of vector dimensions
    int dim = sq.d;

    // set up keys on device
    auto keysDevice = toDeviceTemporary<faiss::idx_t, 1>(
        resources_.get(), config_.device, const_cast<idx_t*>(keys), stream, {n}
    );

    // set up output on device
    auto outDevice = toDeviceTemporary<float, 2>(
        resources_.get(), config_.device, out, stream, {n, dim}
    );

    // make codec
    Codec<ScalarQuantizer::QuantizerType::QT_8bit, 1> codec(
        sq.code_size, gpuTrained.data(), gpuTrained.data() + dim
    );

    // determine block size and layout
    idx_t blockSize = kWarpSize * kScalarQuantizerWarps;
    idx_t numBlocks = utils::divUp(n, blockSize);
    FAISS_ASSERT(n % blockSize == 0);

    // launch kernel
    auto grid = dim3(numBlocks);
    auto block = dim3(blockSize);
    reconstructWithCodec<<<grid, block, codec.getSmemSize(dim), stream>>>(
        codec, n, dim, keysDevice.data(), (void*)gpuCodes.data(), outDevice.data()
    );

    // copy back to host if needed
    fromDevice<float, 2>(outDevice, out, stream);
}

void GpuIndexScalarQuantizer::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {};

void GpuIndexScalarQuantizer::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {};

// not trainable or searchable right now

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
