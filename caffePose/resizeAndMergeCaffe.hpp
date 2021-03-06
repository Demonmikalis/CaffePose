#ifndef OPENPOSE_NET_RESIZE_AND_MERGE_CAFFE_HPP
#define OPENPOSE_NET_RESIZE_AND_MERGE_CAFFE_HPP
#include "config.hpp"
#include "array.hpp"
#include "point.hpp"

// PIMPL does not work here. Alternative:
// stackoverflow.com/questions/13978775/how-to-avoid-include-dependency-to-external-library?answertab=active#tab-top
namespace caffe
{
    template <typename T> class Blob;
}

namespace op
{
    // It mostly follows the Caffe::layer implementation, so Caffe users can easily use it. However, in order to keep
    // the compatibility with any generic Caffe version, we keep this 'layer' inside our library rather than in the
    // Caffe code.
    template <typename T>
    class OP_API ResizeAndMergeCaffe
    {
    public:
        explicit ResizeAndMergeCaffe();

        virtual void LayerSetUp(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top);

        virtual void Reshape(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top,
                             const T netFactor, const T scaleFactor, const bool mergeFirstDimension = true,
                             const int gpuID = 0);

        virtual inline const char* type() const { return "ResizeAndMerge"; }

        void setScaleRatios(const std::vector<T>& scaleRatios);

        virtual void Forward_cpu(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top);

        virtual void Forward_gpu(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top);

        virtual void Forward_ocl(const std::vector<caffe::Blob<T>*>& bottom, const std::vector<caffe::Blob<T>*>& top);

        virtual void Backward_cpu(const std::vector<caffe::Blob<T>*>& top, const std::vector<bool>& propagate_down,
                                  const std::vector<caffe::Blob<T>*>& bottom);

        virtual void Backward_gpu(const std::vector<caffe::Blob<T>*>& top, const std::vector<bool>& propagate_down,
                                  const std::vector<caffe::Blob<T>*>& bottom);

    private:
        std::vector<T*> mTempGPUData;
        std::vector<T> mScaleRatios;
        std::vector<std::array<int, 4>> mBottomSizes;
        std::array<int, 4> mTopSize;
        int mGpuID;

        DELETE_COPY(ResizeAndMergeCaffe);
    };
}

#endif // OPENPOSE_NET_RESIZE_AND_MERGE_CAFFE_HPP
