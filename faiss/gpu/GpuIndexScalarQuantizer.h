// gpu implementation of IndexScalarQuantizer

#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/utils/DeviceTensor.cuh>

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

   protected:
    // on GPU codes
    DeviceTensor<uint8_t, 1, true> gpuCodes;
};

} // namespace gpu
} // namespace faiss
