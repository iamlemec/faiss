// gpu implementation of IndexScalarQuantizer

#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>

namespace faiss {
namespace gpu {

struct GpuIndexScalarQuantizerConfig : public GpuIndexConfig {
    inline GpuIndexScalarQuantizerConfig() {}
};

/// IndexScalarQuantizer on the GPU
class GpuIndexScalarQuantizer : public GpuIndex {
   public:
    /// Construct from a pre-existing faiss::IndexScalarQuantizer instance,
    /// copying data over to the given GPU, if the input index is trained.
    GpuIndexScalarQuantizer(
            GpuResourcesProvider* provider,
            const faiss::IndexScalarQuantizer* index,
            GpuIndexScalarQuantizerConfig config = GpuIndexScalarQuantizerConfig());

    ~GpuIndexScalarQuantizer() override;

    // trains the scalar quantizer based on the given vector data
    void train(idx_t n, const float* x) override;

    // does nothing in this case
    void reset() override;

    // reconstructs a vector from its codes
    void reconstruct_batch(idx_t n, const idx_t* keys, float* recons) const;

   protected:
    bool addImplRequiresIDs_() const override;
    void addImpl_(idx_t n, const float* x, const idx_t* ids) override;
    void searchImpl_(
            idx_t n,
            const float* x,
            int k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override;

    // on GPU quantizer
    GpuScalarQuantizer* gpuSq;

    // on GPU codes
    DeviceTensor<uint8_t, 1, true> gpuCodes;
};

} // namespace gpu
} // namespace faiss
