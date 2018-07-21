#include "resizeAndMergeCaffe.hpp"
#ifdef USE_CAFFE
    #include <caffe/blob.hpp>
#endif
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#ifdef USE_OPENCL
    #include <openpose/gpu/opencl.hcl>
    #include <openpose/gpu/cl2.hpp>
#endif

namespace op
{

    template <typename T>
    void resizeAndMergeCpu(T* targetPtr, const std::vector<const T*>& sourcePtrs,
                           const std::array<int, 4>& targetSize,
                           const std::vector<std::array<int, 4>>& sourceSizes,
                           const std::vector<T>& scaleInputToNetInputs)
    {
        try
        {
            // Scale used in CUDA/CL to know scale ratio between input and output
            // CPU directly uses sourceWidth/Height and targetWidth/Height
            UNUSED(scaleInputToNetInputs);

            // Security checks
            if (sourceSizes.empty())
                error("sourceSizes cannot be empty.", __LINE__, __FUNCTION__, __FILE__);

            // Params
            const auto nums = (signed)sourceSizes.size();
            const auto channels = targetSize[1]; // 57
            const auto targetHeight = targetSize[2]; // 368
            const auto targetWidth = targetSize[3]; // 496
            const auto targetChannelOffset = targetWidth * targetHeight;

            // No multi-scale merging or no merging required
            if (sourceSizes.size() == 1)
            {
                // Params
                const auto& sourceSize = sourceSizes[0];
                const auto sourceHeight = sourceSize[2]; // 368/8 ..
                const auto sourceWidth = sourceSize[3]; // 496/8 ..
                const auto sourceChannelOffset = sourceHeight * sourceWidth;
                if (sourceSize[0] != 1)
                    error("It should never reache this point. Notify us otherwise.",
                          __LINE__, __FUNCTION__, __FILE__);

                // Per channel resize
                const T* sourcePtr = sourcePtrs[0];
                for (auto c = 0 ; c < channels ; c++)
                {
                    cv::Mat source(cv::Size(sourceWidth, sourceHeight), CV_32FC1,
                                   const_cast<T*>(&sourcePtr[c*sourceChannelOffset]));
                    cv::Mat target(cv::Size(targetWidth, targetHeight), CV_32FC1,
                                   (&targetPtr[c*targetChannelOffset]));
                    cv::resize(source, target, {targetWidth, targetHeight}, 0, 0, CV_INTER_CUBIC);
                }
            }
            // Multi-scale merging
            else
            {
                // Construct temp targets. We resuse targetPtr to store first scale
                std::vector<std::unique_ptr<T>> tempTargetPtrs;
                for (auto n = 1; n < nums; n++){
                    tempTargetPtrs.emplace_back(std::unique_ptr<T>(new T[targetChannelOffset * channels]()));
                }

                // Resize and sum
                for (auto n = 0; n < nums; n++){

                    // Params
                    const auto& sourceSize = sourceSizes[n];
                    const auto sourceHeight = sourceSize[2]; // 368/6 ..
                    const auto sourceWidth = sourceSize[3]; // 496/8 ..
                    const auto sourceChannelOffset = sourceHeight * sourceWidth;

                    // Access pointers
                    const T* sourcePtr = sourcePtrs[n];
                    T* tempTargetPtr;
                    if (n != 0)
                        tempTargetPtr = tempTargetPtrs[n-1].get();
                    else
                        tempTargetPtr = targetPtr;

                    T* firstTempTargetPtr = targetPtr;
                    for (auto c = 0 ; c < channels ; c++)
                    {
                        // Resize
                        cv::Mat source(cv::Size(sourceWidth, sourceHeight), CV_32FC1,
                                       const_cast<T*>(&sourcePtr[c*sourceChannelOffset]));
                        cv::Mat target(cv::Size(targetWidth, targetHeight), CV_32FC1,
                                       (&tempTargetPtr[c*targetChannelOffset]));
                        cv::resize(source, target, {targetWidth, targetHeight}, 0, 0, CV_INTER_CUBIC);

                        // Add
                        if (n != 0)
                        {
                            cv::Mat addTarget(cv::Size(targetWidth, targetHeight), CV_32FC1,
                                              (&firstTempTargetPtr[c*targetChannelOffset]));
                            cv::add(target, addTarget, addTarget);
                        }
                    }
                }

                // Average
                for (auto c = 0 ; c < channels ; c++)
                {
                    cv::Mat target(cv::Size(targetWidth, targetHeight), CV_32FC1, (&targetPtr[c*targetChannelOffset]));
                    target /= (float)nums;
                }

            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    ResizeAndMergeCaffe<T>::ResizeAndMergeCaffe() :
        mScaleRatios{T(1)}
    {
        try
        {
            #ifndef USE_CAFFE
                error("OpenPose must be compiled with the `USE_CAFFE` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::LayerSetUp(const std::vector<caffe::Blob<T>*>& bottom,
                                            const std::vector<caffe::Blob<T>*>& top)
    {
        try
        {
            #ifdef USE_CAFFE
                if (top.size() != 1)
                    error("top.size() != 1.", __LINE__, __FUNCTION__, __FILE__);
                if (bottom.size() != 1)
                    error("bottom.size() != 1.", __LINE__, __FUNCTION__, __FILE__);
            #else
                UNUSED(bottom);
                UNUSED(top);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::Reshape(const std::vector<caffe::Blob<T>*>& bottom,
                                         const std::vector<caffe::Blob<T>*>& top,
                                         const T netFactor,
                                         const T scaleFactor,
                                         const bool mergeFirstDimension,
                                         const int gpuID)
    {
        try
        {
            #ifdef USE_CAFFE
                // Security checks
                if (top.size() != 1)
                    error("top.size() != 1", __LINE__, __FUNCTION__, __FILE__);
                if (bottom.empty())
                    error("bottom cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
                // Data
                auto* topBlob = top.at(0);
                const auto* bottomBlob = bottom.at(0);
                // Set top shape
                auto topShape = bottomBlob->shape();
                topShape[0] = (mergeFirstDimension ? 1 : bottomBlob->shape(0));
                // -1 and later +1 to take into account that we are using 0-based index
                // E.g. 100x100 image --> 200x200 --> 0-99 to 0-199 --> scale = 199/99 (not 2!)
                // E.g. 101x101 image --> 201x201 --> scale = 2
                // Test: pixel 0 --> 0, pixel 99 (ex 1) --> 199, pixel 100 (ex 2) --> 200
                topShape[2] = intRound((topShape[2]*netFactor - 1.f) * scaleFactor) + 1;
                topShape[3] = intRound((topShape[3]*netFactor - 1.f) * scaleFactor) + 1;
                topBlob->Reshape(topShape);
                // Array sizes
                mTopSize = std::array<int, 4>{topBlob->shape(0), topBlob->shape(1), topBlob->shape(2),
                                              topBlob->shape(3)};
                mBottomSizes.resize(bottom.size());
                for (auto i = 0u ; i < mBottomSizes.size() ; i++)
                    mBottomSizes[i] = std::array<int, 4>{bottom[i]->shape(0), bottom[i]->shape(1),
                                                         bottom[i]->shape(2), bottom[i]->shape(3)};
                #ifdef USE_OPENCL
                    // GPU ID
                    mGpuID = gpuID;
                    mTempGPUData.resize(mBottomSizes.size(), nullptr);
                #else
                    UNUSED(gpuID);
                #endif
            #else
                UNUSED(bottom);
                UNUSED(top);
                UNUSED(netFactor);
                UNUSED(scaleFactor);
                UNUSED(mergeFirstDimension);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::setScaleRatios(const std::vector<T>& scaleRatios)
    {
        try
        {
            mScaleRatios = {scaleRatios};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::Forward_cpu(const std::vector<caffe::Blob<T>*>& bottom,
                                             const std::vector<caffe::Blob<T>*>& top)
    {
        try
        {
            #ifdef USE_CAFFE
                std::vector<const T*> sourcePtrs(bottom.size());
                for (auto i = 0u ; i < sourcePtrs.size() ; i++)
                    sourcePtrs[i] = bottom[i]->cpu_data();
                resizeAndMergeCpu(top.at(0)->mutable_cpu_data(), sourcePtrs, mTopSize, mBottomSizes,
                                  mScaleRatios);
            #else
                UNUSED(bottom);
                UNUSED(top);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::Forward_gpu(const std::vector<caffe::Blob<T>*>& bottom,
                                             const std::vector<caffe::Blob<T>*>& top)
    {
        try
        {
            #if defined USE_CAFFE && defined USE_CUDA
                std::vector<const T*> sourcePtrs(bottom.size());
                for (auto i = 0u ; i < sourcePtrs.size() ; i++)
                    sourcePtrs[i] = bottom[i]->gpu_data();
                resizeAndMergeGpu(top.at(0)->mutable_gpu_data(), sourcePtrs, mTopSize, mBottomSizes,
                                  mScaleRatios);
            #else
                UNUSED(bottom);
                UNUSED(top);
                error("OpenPose must be compiled with the `USE_CAFFE` & `USE_CUDA` macro definitions in order to run"
                      " this functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::Forward_ocl(const std::vector<caffe::Blob<T>*>& bottom,
                                             const std::vector<caffe::Blob<T>*>& top)
    {
        try
        {
            #if defined USE_CAFFE && defined USE_OPENCL
                std::vector<const T*> sourcePtrs(bottom.size());
                for (auto i = 0u ; i < sourcePtrs.size() ; i++)
                    sourcePtrs[i] = bottom[i]->gpu_data();
                resizeAndMergeOcl(top.at(0)->mutable_gpu_data(), sourcePtrs, mTempGPUData, mTopSize, mBottomSizes,
                                  mScaleRatios, mGpuID);
            #else
                UNUSED(bottom);
                UNUSED(top);
                error("OpenPose must be compiled with the `USE_CAFFE` & `USE_OPENCL` macro definitions in order to run"
                      " this functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::Backward_cpu(const std::vector<caffe::Blob<T>*>& top,
                                              const std::vector<bool>& propagate_down,
                                              const std::vector<caffe::Blob<T>*>& bottom)
    {
        
        try
        {
            UNUSED(top);
            UNUSED(propagate_down);
            UNUSED(bottom);
            #ifdef USE_CAFFE
                NOT_IMPLEMENTED;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
        
    }

    template <typename T>
    void ResizeAndMergeCaffe<T>::Backward_gpu(const std::vector<caffe::Blob<T>*>& top,
                                              const std::vector<bool>& propagate_down,
                                              const std::vector<caffe::Blob<T>*>& bottom)
    {
        
        try
        {
            UNUSED(top);
            UNUSED(propagate_down);
            UNUSED(bottom);
            #ifdef USE_CAFFE
                NOT_IMPLEMENTED;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_FLOATING_TYPES_CLASS(ResizeAndMergeCaffe);
}
