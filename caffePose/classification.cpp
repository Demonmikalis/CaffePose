#include "config.hpp"
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "array.hpp"
#include "point.hpp"
#include "rectangle.hpp"
#include "netCaffe.hpp"
#include "nmsCaffe.hpp"
#include "bodyPartConnectorCaffe.hpp"
#include "resizeAndMergeCaffe.hpp"
#include <memory>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using namespace op;
using namespace std;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 3);

 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  boost::shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

/* ------------------------- Data Prepare -------------------------*/

void unrollArrayToUCharCvMat(cv::Mat& cvMatResult, const Array<float>& array)
{
    try
    {
        if (array.getNumberDimensions() != 3)
            error("Only implemented for array.getNumberDimensions() == 3 so far.",
                  __LINE__, __FUNCTION__, __FILE__);

        if (!array.empty())
        {
            const auto channels = array.getSize(0);
            const auto height = array.getSize(1);
            const auto width = array.getSize(2);
            const auto areaInput = height * width;
            const auto areaOutput = channels * width;
            // Allocate cv::Mat if it was not initialized yet
            if (cvMatResult.empty() || cvMatResult.cols != channels * width || cvMatResult.rows != height)
                cvMatResult = cv::Mat(height, areaOutput, CV_8UC1);
            // Fill cvMatResult from array
            for (auto channel = 0 ; channel < channels ; channel++)
            {
                // Get memory to be modified
                cv::Mat cvMatROI(cvMatResult, cv::Rect{channel * width, 0, width, height});
                // Modify memory
                const auto* arrayPtr = array.getConstPtr() + channel * areaInput;
                for (auto y = 0 ; y < height ; y++)
                {
                    auto* cvMatROIPtr = cvMatROI.ptr<uchar>(y);
                    const auto offsetHeight = y * width;
                    for (auto x = 0 ; x < width ; x++)
                    {
                        const auto value = uchar( fastTruncate(intRound(arrayPtr[offsetHeight + x]), 0, 255) );
                        cvMatROIPtr[x] = (unsigned char)(value);
                    }
                }
            }
        }
        else
            cvMatResult = cv::Mat();
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

void uCharCvMatToFloatPtr(float* floatPtrImage, const cv::Mat& cvImage, const int normalize)
{
    try
    {
        // float* (deep net format): C x H x W
        // cv::Mat (OpenCV format): H x W x C
        const int width = cvImage.cols;
        const int height = cvImage.rows;
        const int channels = cvImage.channels();

        const auto* const originFramePtr = cvImage.data;    // cv::Mat.data is always uchar
        for (auto c = 0; c < channels; c++)
        {
            const auto floatPtrImageOffsetC = c * height;
            for (auto y = 0; y < height; y++)
            {
                const auto floatPtrImageOffsetY = (floatPtrImageOffsetC + y) * width;
                const auto originFramePtrOffsetY = y * width;
                for (auto x = 0; x < width; x++)
                    floatPtrImage[floatPtrImageOffsetY + x] = float(originFramePtr[(originFramePtrOffsetY + x)
                                                                    * channels + c]);
            }
        }
        // Normalizing if desired
        // floatPtrImage wrapped as cv::Mat
            // Empirically tested - OpenCV is more efficient normalizing a whole matrix/image (it uses AVX and
            // other optimized instruction sets).
            // In addition, the following if statement does not copy the pointer to a cv::Mat, just wrapps it.
        // VGG
        if (normalize == 1)
        {
            cv::Mat floatPtrImageCvWrapper(height, width, CV_32FC3, floatPtrImage);
            floatPtrImageCvWrapper = floatPtrImageCvWrapper/256.f - 0.5f;
        }
        // // ResNet
        // else if (normalize == 2)
        // {
        //     const int imageArea = width * height;
        //     const std::array<float,3> means{102.9801, 115.9465, 122.7717};
        //     for (auto i = 0 ; i < 3 ; i++)
        //     {
        //         cv::Mat floatPtrImageCvWrapper(height, width, CV_32FC1, floatPtrImage + i*imageArea);
        //         floatPtrImageCvWrapper = floatPtrImageCvWrapper - means[i];
        //     }
        // }
        // DenseNet
        else if (normalize == 2)
        {
            const auto scaleDenseNet = 0.017;
            const int imageArea = width * height;
            const std::array<float,3> means{103.94,116.78,123.68};
            for (auto i = 0 ; i < 3 ; i++)
            {
                cv::Mat floatPtrImageCvWrapper(height, width, CV_32FC1, floatPtrImage + i*imageArea);
                floatPtrImageCvWrapper = scaleDenseNet*(floatPtrImageCvWrapper - means[i]);
            }
        }
        // Unknown
        else if (normalize != 0)
            error("Unknown normalization value (" + to_string(normalize) + ").",
                  __LINE__, __FUNCTION__, __FILE__);
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

double resizeGetScaleFactor(const Point<int>& initialSize, const Point<int>& targetSize)
{
    try
    {
        const auto ratioWidth = (targetSize.x - 1) / (double)(initialSize.x - 1);
        const auto ratioHeight = (targetSize.y - 1) / (double)(initialSize.y - 1);
        return fastMin(ratioWidth, ratioHeight);
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return 0.;
    }
}

cv::Mat resizeFixedAspectRatio(const cv::Mat& cvMat, const double scaleFactor, const Point<int>& targetSize,
                               const int borderMode, const cv::Scalar& borderValue)
{
    try
    {
        const cv::Size cvTargetSize{targetSize.x, targetSize.y};
        cv::Mat resultingCvMat;
        cv::Mat M = cv::Mat::eye(2,3,CV_64F);
        M.at<double>(0,0) = scaleFactor;
        M.at<double>(1,1) = scaleFactor;
        if (scaleFactor != 1. || cvTargetSize != cvMat.size())
            cv::warpAffine(cvMat, resultingCvMat, M, cvTargetSize,
                           (scaleFactor < 1. ? cv::INTER_AREA : cv::INTER_CUBIC), borderMode, borderValue);
        else
            resultingCvMat = cvMat.clone();
        return resultingCvMat;
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return cv::Mat();
    }
}


std::vector<Array<float>> CvMatToOpInput_createArray(const cv::Mat& cvInputData,
                                                      const std::vector<double>& scaleInputToNetInputs,
                                                      const std::vector<Point<int>>& netInputSizes)
{
    try
    {
        // Security checks
        if (cvInputData.empty())
            error("Wrong input element (empty cvInputData).", __LINE__, __FUNCTION__, __FILE__);
        if (cvInputData.channels() != 3)
            error("Input images must be 3-channel BGR.", __LINE__, __FUNCTION__, __FILE__);
        if (scaleInputToNetInputs.size() != netInputSizes.size())
            error("scaleInputToNetInputs.size() != netInputSizes.size().", __LINE__, __FUNCTION__, __FILE__);
        // inputNetData - Reescale keeping aspect ratio and transform to float the input deep net image
        const auto numberScales = (int)scaleInputToNetInputs.size();
        std::vector<Array<float>> inputNetData(numberScales);
        for (auto i = 0u ; i < inputNetData.size() ; i++)
        {
            inputNetData[i].reset({1, 3, netInputSizes.at(i).y, netInputSizes.at(i).x});
            std::vector<double> scaleRatios(numberScales, 1.f);
            const cv::Mat frameWithNetSize = resizeFixedAspectRatio(cvInputData, scaleInputToNetInputs[i],
                                                                    netInputSizes[i],cv::BORDER_CONSTANT,
                                                                    cv::Scalar{0,0,0});
            // Fill inputNetData[i]
            //uCharCvMatToFloatPtr(inputNetData[i].getPtr(), frameWithNetSize,
                                 //(mPoseModel == PoseModel::BODY_19N ? 2 : 1));
            uCharCvMatToFloatPtr(inputNetData[i].getPtr(), frameWithNetSize,1);
        }
        return inputNetData;
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return {};
    }
}

Array<float> CvMatToOpOutput_createArray(const cv::Mat& cvInputData, const double scaleInputToOutput,
                                              const Point<int>& outputResolution)
{
    try
    {
        // Security checks
        if (cvInputData.empty())
            error("Wrong input element (empty cvInputData).", __LINE__, __FUNCTION__, __FILE__);
        if (cvInputData.channels() != 3)
            error("Input images must be 3-channel BGR.", __LINE__, __FUNCTION__, __FILE__);
        if (cvInputData.cols <= 0 || cvInputData.rows <= 0)
            error("Input images has 0 area.", __LINE__, __FUNCTION__, __FILE__);
        if (outputResolution.x <= 0 || outputResolution.y <= 0)
            error("Output resolution has 0 area.", __LINE__, __FUNCTION__, __FILE__);
        // outputData - Reescale keeping aspect ratio and transform to float the output image
        const cv::Mat frameWithOutputSize = resizeFixedAspectRatio(cvInputData, scaleInputToOutput,
                                                                   outputResolution,cv::BORDER_CONSTANT,
                                                                    cv::Scalar{0,0,0});
        Array<float> outputData({outputResolution.y, outputResolution.x, 3});
        frameWithOutputSize.convertTo(outputData.getCvMat(), CV_32FC3);
        // Return result
        return outputData;
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return Array<float>{};
    }
}

std::tuple<std::vector<double>, std::vector<Point<int>>, double, Point<int>> ScaleAndSizeExtractor_extract(
    const Point<int>& inputResolution,Point<int> mNetInputResolution, Point<int> mOutputSize,
    int mScaleNumber,
    double mScaleGap)
{
    try
    {
        // Security checks
        if (inputResolution.area() <= 0)
            error("Wrong input element (empty cvInputData).", __LINE__, __FUNCTION__, __FILE__);
        // Set poseNetInputSize
        auto poseNetInputSize = mNetInputResolution;
        if (poseNetInputSize.x <= 0 || poseNetInputSize.y <= 0)
        {
            // Security checks
            if (poseNetInputSize.x <= 0 && poseNetInputSize.y <= 0)
                error("Only 1 of the dimensions of net input resolution can be <= 0.",
                      __LINE__, __FUNCTION__, __FILE__);
            if (poseNetInputSize.x <= 0)
                poseNetInputSize.x = 16 * intRound(
                    poseNetInputSize.y * inputResolution.x / (float) inputResolution.y / 16.f
                );
            else // if (poseNetInputSize.y <= 0)
                poseNetInputSize.y = 16 * intRound(
                    poseNetInputSize.x * inputResolution.y / (float) inputResolution.x / 16.f
                );
        }
        // scaleInputToNetInputs & netInputSizes - Reescale keeping aspect ratio
        std::vector<double> scaleInputToNetInputs(mScaleNumber, 1.f);
        std::vector<Point<int>> netInputSizes(mScaleNumber);
        for (auto i = 0; i < mScaleNumber; i++)
        {
            const auto currentScale = 1. - i*mScaleGap;
            if (currentScale < 0. || 1. < currentScale)
                error("All scales must be in the range [0, 1], i.e. 0 <= 1-scale_number*scale_gap <= 1",
                      __LINE__, __FUNCTION__, __FILE__);

            const auto targetWidth = fastTruncate(intRound(poseNetInputSize.x * currentScale) / 16 * 16, 1,
                                                  poseNetInputSize.x);
            const auto targetHeight = fastTruncate(intRound(poseNetInputSize.y * currentScale) / 16 * 16, 1,
                                                   poseNetInputSize.y);
            const Point<int> targetSize{targetWidth, targetHeight};
            scaleInputToNetInputs[i] = resizeGetScaleFactor(inputResolution, targetSize);
            netInputSizes[i] = targetSize;
        }
        // scaleInputToOutput - Scale between input and desired output size
        Point<int> outputResolution;
        double scaleInputToOutput;
        // Output = mOutputSize3D size
        if (mOutputSize.x > 0 && mOutputSize.y > 0)
        {
            outputResolution = mOutputSize;
            scaleInputToOutput = resizeGetScaleFactor(inputResolution, outputResolution);
        }
        // Output = input size
        else
        {
            outputResolution = inputResolution;
            scaleInputToOutput = 1.;
        }
        // Return result
        return std::make_tuple(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution);
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return std::make_tuple(std::vector<double>{}, std::vector<Point<int>>{}, 1., Point<int>{});
    }
}


/* ------------------------- Caffe Network Core -------------------------*/
struct ImplPoseExtractorCaffe
{
  // Used when increasing spCaffeNets
  const PoseModel mPoseModel;
  const int mGpuId;
  const std::string mModelFolder;
  const bool mEnableGoogleLogging;
  // General parameters
  std::vector<std::shared_ptr<NetCaffe>> spCaffeNets;
  std::shared_ptr<ResizeAndMergeCaffe<float>> spResizeAndMergeCaffe;
  std::shared_ptr<NmsCaffe<float>> spNmsCaffe;
  std::shared_ptr<BodyPartConnectorCaffe<float>> spBodyPartConnectorCaffe;
  std::vector<std::vector<int>> mNetInput4DSizes;
  std::vector<double> mScaleInputToNetInputs;
  // Init with thread
  std::vector<boost::shared_ptr<caffe::Blob<float>>> spCaffeNetOutputBlobs;
  std::shared_ptr<caffe::Blob<float>> spHeatMapsBlob;
  std::shared_ptr<caffe::Blob<float>> spPeaksBlob;

  ImplPoseExtractorCaffe(const PoseModel poseModel, const int gpuId,
                         const std::string& modelFolder, const bool enableGoogleLogging) :
      mPoseModel{poseModel},
      mGpuId{gpuId},
      mModelFolder{modelFolder},
      mEnableGoogleLogging{enableGoogleLogging},
      spResizeAndMergeCaffe{std::make_shared<ResizeAndMergeCaffe<float>>()},
      spNmsCaffe{std::make_shared<NmsCaffe<float>>()},
      spBodyPartConnectorCaffe{std::make_shared<BodyPartConnectorCaffe<float>>()}
  {
  }

};

struct ImplPoseExtractorCaffe *upImpl;

static void initUpImpl(const PoseModel poseModel, const std::string& modelFolder,
  const int gpuId, const std::vector<HeatMapType>& heatMapTypes,
  const ScaleMode heatMapScale, const bool addPartCandidates,
  const bool enableGoogleLogging)
{
  upImpl = new ImplPoseExtractorCaffe(poseModel, gpuId, 
    modelFolder, enableGoogleLogging);
  upImpl->spBodyPartConnectorCaffe->setPoseModel(upImpl->mPoseModel);
}

static void addCaffeNetOnThread(std::vector<std::shared_ptr<NetCaffe>>& netCaffe,
                         std::vector<boost::shared_ptr<caffe::Blob<float>>>& caffeNetOutputBlob,
                         const PoseModel poseModel, const int gpuId,
                         const std::string& modelFolder, const bool enableGoogleLogging)
{
    try
    {
        std::string strPoseProtoTxt="pose/body_25/pose_deploy.prototxt";
        std::string strPoseTrainedModel="pose/body_25/pose_iter_584000.caffemodel";
        // Add Caffe Net
        netCaffe.emplace_back(
            //std::make_shared<NetCaffe>(modelFolder + getPoseProtoTxt(poseModel),
            //                           modelFolder + getPoseTrainedModel(poseModel),
            //                           gpuId, enableGoogleLogging)
            std::make_shared<NetCaffe>(modelFolder + strPoseProtoTxt,
                                       modelFolder + strPoseTrainedModel,
                                       gpuId, enableGoogleLogging)
        );
        // Initializing them on the thread
        netCaffe.back()->initializationOnThread();
        caffeNetOutputBlob.emplace_back(netCaffe.back()->getOutputBlob());
        // Security checks
        if (netCaffe.size() != caffeNetOutputBlob.size())
            error("Weird error, this should not happen. Notify us.", __LINE__, __FUNCTION__, __FILE__);
        // Cuda check
        #ifdef USE_CUDA
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        #endif
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

static std::vector<caffe::Blob<float>*> caffeNetSharedToPtr(
            std::vector<boost::shared_ptr<caffe::Blob<float>>>& caffeNetOutputBlob)
{
    try
    {
        // Prepare spCaffeNetOutputBlobss
        std::vector<caffe::Blob<float>*> caffeNetOutputBlobs(caffeNetOutputBlob.size());
        for (auto i = 0u ; i < caffeNetOutputBlobs.size() ; i++)
            caffeNetOutputBlobs[i] = caffeNetOutputBlob[i].get();
        return caffeNetOutputBlobs;
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return {};
    }
}

const std::array<float, (int)PoseModel::Size> POSE_CCN_DECREASE_FACTOR{
      8.f,    // BODY_25
      8.f,    // COCO
      8.f,    // MPI_15
      8.f,    // MPI_15_4
      8.f,    // BODY_19
      4.f,    // BODY_19_X2
      8.f,    // BODY_59
      8.f,    // BODY_19N
      8.f,    // BODY_19b
      8.f,    // BODY_25_19
      8.f,    // BODY_65
      8.f,    // CAR_12
};
static float getPoseNetDecreaseFactor(const PoseModel poseModel)
{
    try
    {
        return POSE_CCN_DECREASE_FACTOR.at((int)poseModel);
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return 0.f;
    }
}
static unsigned int getPoseNumberBodyParts(const PoseModel poseModel)
{
    const std::array<unsigned int, (int)PoseModel::Size> 
    POSE_NUMBER_BODY_PARTS{
        25, 18, 15, 15, 19, 19, 59, 19, 19, 25, 65, 12
    };
    try
    {
        return POSE_NUMBER_BODY_PARTS.at((int)poseModel);
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return 0u;
    }
}
static unsigned int getPoseMaxPeaks(const PoseModel poseModel)
{
    try
    {
        const auto mPOSE_MAX_PEOPLE = 127u;
        //return POSE_MAX_PEAKS.at((int)poseModel);
        return mPOSE_MAX_PEOPLE;
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return 0u;
    }
}

static void reshapePoseExtractorCaffe(
    std::shared_ptr<ResizeAndMergeCaffe<float>>& resizeAndMergeCaffe,
    std::shared_ptr<NmsCaffe<float>>& nmsCaffe,
    std::shared_ptr<BodyPartConnectorCaffe<float>>& bodyPartConnectorCaffe,
    std::vector<boost::shared_ptr<caffe::Blob<float>>>& caffeNetOutputBlob,
    std::shared_ptr<caffe::Blob<float>>& heatMapsBlob,
    std::shared_ptr<caffe::Blob<float>>& peaksBlob,
    const float scaleInputToNetInput,
    const PoseModel poseModel,
    const int gpuID)
{
    try
    {
        // HeatMaps extractor blob and layer
        // Caffe modifies bottom - Heatmap gets resized
        const auto caffeNetOutputBlobs = caffeNetSharedToPtr(caffeNetOutputBlob);
        resizeAndMergeCaffe->Reshape(caffeNetOutputBlobs, {heatMapsBlob.get()},
                                     getPoseNetDecreaseFactor(poseModel), 1.f/scaleInputToNetInput, true,
                                     gpuID);
        // Pose extractor blob and layer
        nmsCaffe->Reshape({heatMapsBlob.get()}, {peaksBlob.get()}, getPoseMaxPeaks(poseModel),
                          getPoseNumberBodyParts(poseModel), gpuID);
        // Pose extractor blob and layer
        bodyPartConnectorCaffe->Reshape({heatMapsBlob.get(), peaksBlob.get()});
        // Cuda check
        #ifdef USE_CUDA
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        #endif
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}



void PoseExtractorCaffe_netInitializationOnThread()
{
    try
    {
        #ifdef USE_CAFFE
            // Logging
            log("Starting initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            // Initialize Caffe net
            addCaffeNetOnThread(upImpl->spCaffeNets, upImpl->spCaffeNetOutputBlobs, upImpl->mPoseModel,
                                upImpl->mGpuId, upImpl->mModelFolder, upImpl->mEnableGoogleLogging);
            #ifdef USE_CUDA
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            #endif
            // Initialize blobs
            upImpl->spHeatMapsBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
            upImpl->spPeaksBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
            #ifdef USE_CUDA
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            #endif
            // Logging
            log("Finished initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        #endif
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

void PoseExtractorCaffe_forwardPass(const std::vector<Array<float>>& inputNetData,
                                   const Point<int>& inputDataSize,
                                   const std::vector<double>& scaleInputToNetInputs,
                                   Array<float> &mPoseKeypoints)
{
  try
  {
      #ifdef USE_CAFFE
          
          Array<float> mPoseScores;
          float mScaleNetToOutput;
          Point<int> mNetOutputSize;
          // Security checks
          if (inputNetData.empty())
              error("Empty inputNetData.", __LINE__, __FUNCTION__, __FILE__);
          for (const auto& inputNetDataI : inputNetData)
              if (inputNetDataI.empty())
                  error("Empty inputNetData.", __LINE__, __FUNCTION__, __FILE__);
          if (inputNetData.size() != scaleInputToNetInputs.size())
              error("Size(inputNetData) must be same than size(scaleInputToNetInputs).",
                    __LINE__, __FUNCTION__, __FILE__);

          // Resize std::vectors if required
          const auto numberScales = inputNetData.size();
          upImpl->mNetInput4DSizes.resize(numberScales);
          while (upImpl->spCaffeNets.size() < numberScales)
              addCaffeNetOnThread(upImpl->spCaffeNets, upImpl->spCaffeNetOutputBlobs, upImpl->mPoseModel,
                                  upImpl->mGpuId, upImpl->mModelFolder, false);

          // std::cout << inputNetData.size() << std::endl;
          // Process each image
          for (auto i = 0u ; i < inputNetData.size(); i++)
          {
              // 1. Caffe deep network
              upImpl->spCaffeNets.at(i)->forwardPass(inputNetData[i]);                                   // ~80ms

              // Reshape blobs if required
              // Note: In order to resize to input size to have same results as Matlab, uncomment the commented
              // lines
              // Note: For dynamic sizes (e.g. a folder with images of different aspect ratio)
              if (!vectorsAreEqual(upImpl->mNetInput4DSizes.at(i), inputNetData[i].getSize()))
                  // || !vectorsAreEqual(upImpl->mScaleInputToNetInputs, scaleInputToNetInputs))
              {
                  upImpl->mNetInput4DSizes.at(i) = inputNetData[i].getSize();
                  mNetOutputSize = Point<int>{upImpl->mNetInput4DSizes[0][3],
                                              upImpl->mNetInput4DSizes[0][2]};
                  
                  // upImpl->mScaleInputToNetInputs = scaleInputToNetInputs;
                  
                  reshapePoseExtractorCaffe(upImpl->spResizeAndMergeCaffe, upImpl->spNmsCaffe,
                                            upImpl->spBodyPartConnectorCaffe, upImpl->spCaffeNetOutputBlobs,
                                            upImpl->spHeatMapsBlob, upImpl->spPeaksBlob,
                                            1.f, upImpl->mPoseModel, upImpl->mGpuId);
                                            // scaleInputToNetInputs[i], upImpl->mPoseModel);
                  printf("reshape passed!\n");
                  
              }
          }
          
          // 2. Resize heat maps + merge different scales
          const auto caffeNetOutputBlobs = caffeNetSharedToPtr(upImpl->spCaffeNetOutputBlobs);
          const std::vector<float> floatScaleRatios(scaleInputToNetInputs.begin(), scaleInputToNetInputs.end());
          upImpl->spResizeAndMergeCaffe->setScaleRatios(floatScaleRatios);

          #ifdef USE_CUDA
              //upImpl->spResizeAndMergeCaffe->Forward_cpu(caffeNetOutputBlobs, {upImpl->spHeatMapsBlob.get()}); // ~20ms
              upImpl->spResizeAndMergeCaffe->Forward_gpu(caffeNetOutputBlobs, {upImpl->spHeatMapsBlob.get()}); // ~5ms
          #elif USE_OPENCL
              //upImpl->spResizeAndMergeCaffe->Forward_cpu(caffeNetOutputBlobs, {upImpl->spHeatMapsBlob.get()}); // ~20ms
              upImpl->spResizeAndMergeCaffe->Forward_ocl(caffeNetOutputBlobs, {upImpl->spHeatMapsBlob.get()});
          #else
              upImpl->spResizeAndMergeCaffe->Forward_cpu(caffeNetOutputBlobs, {upImpl->spHeatMapsBlob.get()}); // ~20ms
          #endif

          // Get scale net to output (i.e. image input)
          // Note: In order to resize to input size, (un)comment the following lines
          const auto scaleProducerToNetInput = resizeGetScaleFactor(inputDataSize, mNetOutputSize);
          const Point<int> netSize{intRound(scaleProducerToNetInput*inputDataSize.x),
                                   intRound(scaleProducerToNetInput*inputDataSize.y)};
          mScaleNetToOutput = {(float)resizeGetScaleFactor(netSize, inputDataSize)};
          // mScaleNetToOutput = 1.f;

          // 3. Get peaks by Non-Maximum Suppression
          //upImpl->spNmsCaffe->setThreshold((float)get(PoseProperty::NMSThreshold));
          //TODO:MR.Black NMSThreshold = 0.04f
          upImpl->spNmsCaffe->setThreshold((float)0.05f);
          const auto nmsOffset = float(0.5/double(mScaleNetToOutput));
          upImpl->spNmsCaffe->setOffset(Point<float>{nmsOffset, nmsOffset});
          #ifdef USE_CUDA
              //upImpl->spNmsCaffe->Forward_cpu({upImpl->spHeatMapsBlob.get()}, {upImpl->spPeaksBlob.get()}); // ~ 7ms
              upImpl->spNmsCaffe->Forward_gpu({upImpl->spHeatMapsBlob.get()}, {upImpl->spPeaksBlob.get()});// ~2ms
              cudaCheck(__LINE__, __FUNCTION__, __FILE__);
          #elif USE_OPENCL
              //upImpl->spNmsCaffe->Forward_cpu({upImpl->spHeatMapsBlob.get()}, {upImpl->spPeaksBlob.get()}); // ~ 7ms
              upImpl->spNmsCaffe->Forward_ocl({upImpl->spHeatMapsBlob.get()}, {upImpl->spPeaksBlob.get()});
          #else
              upImpl->spNmsCaffe->Forward_cpu({upImpl->spHeatMapsBlob.get()}, {upImpl->spPeaksBlob.get()}); // ~ 7ms
          #endif

          // 4. Connecting body parts
          // Get scale net to output (i.e. image input)
          upImpl->spBodyPartConnectorCaffe->setScaleNetToOutput(mScaleNetToOutput);
          /*
          upImpl->spBodyPartConnectorCaffe->setInterMinAboveThreshold(
              (float)get(PoseProperty::ConnectInterMinAboveThreshold)
          );*/
          upImpl->spBodyPartConnectorCaffe->setInterMinAboveThreshold(
              (float)0.95f
          );
          upImpl->spBodyPartConnectorCaffe->setInterThreshold((float)0.05f);
          upImpl->spBodyPartConnectorCaffe->setMinSubsetCnt((int)3);
          upImpl->spBodyPartConnectorCaffe->setMinSubsetScore((float)0.4f);
          /*
          upImpl->spBodyPartConnectorCaffe->setInterThreshold((float)get(PoseProperty::ConnectInterThreshold));
          upImpl->spBodyPartConnectorCaffe->setMinSubsetCnt((int)get(PoseProperty::ConnectMinSubsetCnt));
          upImpl->spBodyPartConnectorCaffe->setMinSubsetScore((float)get(PoseProperty::ConnectMinSubsetScore));
          */

          // CUDA version not implemented yet
          // #ifdef USE_CUDA
          //     upImpl->spBodyPartConnectorCaffe->Forward_gpu({upImpl->spHeatMapsBlob.get(),
          //                                                    upImpl->spPeaksBlob.get()},
          //                                                   mPoseKeypoints, mPoseScores);
          // #else
              upImpl->spBodyPartConnectorCaffe->Forward_cpu({upImpl->spHeatMapsBlob.get(),
                                                             upImpl->spPeaksBlob.get()},
                                                            mPoseKeypoints, mPoseScores);
                               
          // #endif
      #else
          UNUSED(inputNetData);
          UNUSED(inputDataSize);
          UNUSED(scaleInputToNetInputs);
      #endif

      
  }
  catch (const std::exception& e)
  {
      error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}

/*
class Integer  
{  
    int n;  
public:  
    Integer(int n) : n(n) { }  
    ~Integer() { printf("Deleting %d\n", n); }  
    int get() const { return n; }  
};  
int test_shared_ptr3()  
{  
    auto a = std::make_shared<Integer>(10);  
    auto b = std::make_shared<Integer>(20);  
    auto c = a;  
    auto d = std::make_shared<Integer>(30);  
    auto e = b;  
    a = d;  
    b = std::make_shared<Integer>(40);  
    auto f = c;  
    b = f;  
  
    printf("%d\n", a->get());  
    printf("%d\n", b->get());  
    printf("%d\n", c->get());  
    printf("%d\n", d->get());  
    printf("%d\n", e->get());  
    printf("%d\n", f->get());  
  
    return 0;  
}*/


Rectangle<float> getKeypointsRectangle(const Array<float>& keypoints, const int person, const float threshold)
{
    try
    {
        const auto numberKeypoints = keypoints.getSize(1);
        // Security checks
        if (numberKeypoints < 1)
            error("Number body parts must be > 0", __LINE__, __FUNCTION__, __FILE__);
        // Define keypointPtr
        const auto keypointPtr = keypoints.getConstPtr() + person * keypoints.getSize(1) * keypoints.getSize(2);
        float minX = std::numeric_limits<float>::max();
        float maxX = 0.f;
        float minY = minX;
        float maxY = maxX;
        for (auto part = 0 ; part < numberKeypoints ; part++)
        {
            const auto score = keypointPtr[3*part + 2];
            if (score > threshold)
            {
                const auto x = keypointPtr[3*part];
                const auto y = keypointPtr[3*part + 1];
                // Set X
                if (maxX < x)
                    maxX = x;
                if (minX > x)
                    minX = x;
                // Set Y
                if (maxY < y)
                    maxY = y;
                if (minY > y)
                    minY = y;
            }
        }
        if (maxX >= minX && maxY >= minY)
            return Rectangle<float>{minX, minY, maxX-minX, maxY-minY};
        else
            return Rectangle<float>{};
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return Rectangle<float>{};
    }
}


void renderKeypointsCpu(Array<float>& frameArray, const Array<float>& keypoints,
                        const std::vector<unsigned int>& pairs, const std::vector<float> colors,
                        const float thicknessCircleRatio, const float thicknessLineRatioWRTCircle,
                        const std::vector<float>& poseScales, const float threshold)
{
  #ifdef UNIX_GUI
  try
  {
      if (!frameArray.empty())
      {
          // Array<float> --> cv::Mat
          auto frame = frameArray.getCvMat();

          // Security check
          if (frame.channels() != 3)
              error("errorMessage", __LINE__, __FUNCTION__, __FILE__);

          // Get frame channels
          const auto width = frame.size[1];
          const auto height = frame.size[0];
          const auto area = width * height;
          cv::Mat frameBGR(height, width, CV_32FC3, frame.data);

          // Parameters
          const auto lineType = 8;
          const auto shift = 0;
          const auto numberColors = colors.size();
          const auto numberScales = poseScales.size();
          const auto thresholdRectangle = 0.1f;
          const auto numberKeypoints = keypoints.getSize(1);

          // Keypoints
          for (auto person = 0 ; person < keypoints.getSize(0) ; person++)
          {
              const auto personRectangle = getKeypointsRectangle(keypoints, person, thresholdRectangle);
              if (personRectangle.area() > 0)
              {
                  const auto ratioAreas = fastMin(1.f, fastMax(personRectangle.width/(float)width,
                                                               personRectangle.height/(float)height));
                  // Size-dependent variables
                  const auto thicknessRatio = fastMax(intRound(std::sqrt(area)
                                                               * thicknessCircleRatio * ratioAreas), 2);
                  // Negative thickness in cv::circle means that a filled circle is to be drawn.
                  const auto thicknessCircle = fastMax(1, (ratioAreas > 0.05f ? thicknessRatio : -1));
                  const auto thicknessLine = fastMax(1, intRound(thicknessRatio * thicknessLineRatioWRTCircle));
                  const auto radius = thicknessRatio / 2;

                  // Draw lines
                  for (auto pair = 0u ; pair < pairs.size() ; pair+=2)
                  {
                      const auto index1 = (person * numberKeypoints + pairs[pair]) * keypoints.getSize(2);
                      const auto index2 = (person * numberKeypoints + pairs[pair+1]) * keypoints.getSize(2);
                      if (keypoints[index1+2] > threshold && keypoints[index2+2] > threshold)
                      {
                          const auto thicknessLineScaled = thicknessLine
                                                         * poseScales[pairs[pair+1] % numberScales];
                          const auto colorIndex = pairs[pair+1]*3; // Before: colorIndex = pair/2*3;
                          const cv::Scalar color{
                              colors[(colorIndex+2) % numberColors],
                              colors[(colorIndex+1) % numberColors],
                              colors[colorIndex % numberColors]
                          };
                          const cv::Point keypoint1{intRound(keypoints[index1]), intRound(keypoints[index1+1])};
                          const cv::Point keypoint2{intRound(keypoints[index2]), intRound(keypoints[index2+1])};
                          cv::line(frameBGR, keypoint1, keypoint2, color, thicknessLineScaled, lineType, shift);
                      }
                  }

                  // Draw circles
                  for (auto part = 0 ; part < numberKeypoints ; part++)
                  {
                      const auto faceIndex = (person * numberKeypoints + part) * keypoints.getSize(2);
                      if (keypoints[faceIndex+2] > threshold)
                      {
                          const auto radiusScaled = radius * poseScales[part % numberScales];
                          const auto thicknessCircleScaled = thicknessCircle * poseScales[part % numberScales];
                          const auto colorIndex = part*3;
                          const cv::Scalar color{
                              colors[(colorIndex+2) % numberColors],
                              colors[(colorIndex+1) % numberColors],
                              colors[colorIndex % numberColors]
                          };
                          const cv::Point center{intRound(keypoints[faceIndex]),
                                                 intRound(keypoints[faceIndex+1])};
                          cv::circle(frameBGR, center, radiusScaled, color, thicknessCircleScaled, lineType,
                                     shift);
                      }
                  }
              }
          }
      }
  }
  catch (const std::exception& e)
  {
      error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
  #endif
}

static void scaleKeypoints(Array<float>& keypoints, const float scale)
{
    try
    {
        if (!keypoints.empty() && scale != 1.f)
        {
            // Error check
            if (keypoints.getSize(2) != 3 && keypoints.getSize(2) != 4)
                error("The Array<float> is not a (x,y,score) or (x,y,z,score) format array. This"
                      " function is only for those 2 dimensions: [sizeA x sizeB x 3or4].",
                      __LINE__, __FUNCTION__, __FILE__);
            // Get #people and #parts
            const auto numberPeople = keypoints.getSize(0);
            const auto numberParts = keypoints.getSize(1);
            const auto xyzChannels = keypoints.getSize(2);
            // For each person
            for (auto person = 0 ; person < numberPeople ; person++)
            {
                // For each body part
                for (auto part = 0 ; part < numberParts ; part++)
                {
                    const auto finalIndex = xyzChannels*(person*numberParts + part);
                    for (auto xyz = 0 ; xyz < xyzChannels-1 ; xyz++)
                        keypoints[finalIndex+xyz] *= scale;
                }
            }
        }
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}




void renderPoseKeypointsCpu(Array<float>& frameArray, const Array<float>& poseKeypoints, const PoseModel poseModel,
                                const float renderThreshold, const bool blendOriginalFrame)
{
    try
    {
        if (!frameArray.empty())
        {
            // Background
            if (!blendOriginalFrame)
                frameArray.getCvMat().setTo(0.f); // [0-255]

            // Parameters
            const auto thicknessCircleRatio = 1.f/75.f;
            const auto thicknessLineRatioWRTCircle = 0.75f;
            //const auto& pairs = getPoseBodyPartPairsRender(poseModel);
            //const auto& poseScales = getPoseScales(poseModel);
            const std::vector<unsigned int> pairs = 
              std::vector<unsigned int>{POSE_BODY_25_PAIRS_RENDER_GPU};
            const std::vector<float> poseScales = 
              std::vector<float>{POSE_BODY_25_SCALES_RENDER_GPU}; 
            
            const std::vector<float> poseColors = 
              std::vector<float>{POSE_BODY_25_COLORS_RENDER_GPU};
            // Render keypoints
            renderKeypointsCpu(frameArray, poseKeypoints, pairs, poseColors, thicknessCircleRatio,
                               thicknessLineRatioWRTCircle, poseScales, renderThreshold);
        }
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}


static std::pair<int, std::string> renderPose(Array<float>& outputData,
        const Array<float>& poseKeypoints,
        const float scaleInputToOutput,
        const float scaleNetToOutput)
{
    try
    {
        // Security checks
        if (outputData.empty())
            error("Empty Array<float> outputData.", __LINE__, __FUNCTION__, __FILE__);
        // CPU rendering
        //const auto elementRendered = spElementToRender->load();
        int elementRendered = 0;
        std::string elementRenderedName;
        // Draw poseKeypoints
        if (elementRendered == 0)
        {
            // Rescale keypoints to output size
            auto poseKeypointsRescaled = poseKeypoints.clone();
            scaleKeypoints(poseKeypointsRescaled, scaleInputToOutput);
            // Render keypoints
            //(float)FLAGS_render_threshold, !FLAGS_disable_blending,
            //(float)FLAGS_alpha_pose = 0.05, 1. 0.6
            PoseModel mPoseModel = PoseModel::BODY_25;
            const float mRenderThreshold = 0.05;
            const bool mBlendOriginalFrame = true;
            renderPoseKeypointsCpu(outputData, poseKeypointsRescaled, mPoseModel, mRenderThreshold,
                                   mBlendOriginalFrame);
        }
        // Draw heat maps / PAFs
        else
        {
            UNUSED(scaleNetToOutput);
            error("CPU rendering only available for drawing keypoints, no heat maps nor PAFs.",
                  __LINE__, __FUNCTION__, __FILE__);
        }
        // Return result
        return std::make_pair(elementRendered, elementRenderedName);
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return std::make_pair(-1, "");
    }
}

cv::Mat formatToCvMat(const Array<float>& outputData)
{
    try
    {
        // Security checks
        if (outputData.empty())
            error("Wrong input element (empty outputData).", __LINE__, __FUNCTION__, __FILE__);
        // outputData to cvMat
        cv::Mat cvMat;
        outputData.getConstCvMat().convertTo(cvMat, CV_8UC3);
        // Return cvMat
        return cvMat;
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return cv::Mat();
    }
}

int main(int argc, char** argv) {
  /*  
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg" << std::endl;
    return 1;
  }*/
  
  ::google::InitGoogleLogging("OpenPose");

  string model_file   = "";
  string trained_file = "";
  string mean_file    = "";
  string label_file   = "";
  
  model_file = "./imagenet/deploy.prototxt";
  trained_file = "./imagenet/bvlc_reference_caffenet.caffemodel";
  mean_file = "./imagenet/imagenet_mean.binaryproto";
  label_file = "./imagenet/synset_words.txt";
  
  //Classifier classifier(model_file, trained_file, mean_file, label_file);

  string file = "imagenet/COCO_val2014_000000000192.jpg";

  std::cout << "---------- PoseDetection for "
            << file << " ----------" << std::endl;

  // Step 1 - load image
  cv::Mat inputImage = cv::imread(file, CV_LOAD_IMAGE_COLOR);
  CHECK(!inputImage.empty()) << "Unable to decode image " << file;


  Point<int> outputSize = Point<int>{-1,-1};
  Point<int> netInputSize = Point<int>{-1,80};
  int FLAGS_scale_number = 1;
  double FLAGS_scale_gap = 0.3;

  const Point<int> imageSize{inputImage.cols, inputImage.rows};
  // Step 2 - Get desired scale sizes
  std::vector<double> scaleInputToNetInputs;
  std::vector<op::Point<int>> netInputSizes;
  double scaleInputToOutput;
  Point<int> outputResolution;


  //scale and size extract
  //scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution
  std::tuple<std::vector<double>, std::vector<Point<int>>, double, Point<int>> tp;
  tp = ScaleAndSizeExtractor_extract(imageSize,netInputSize,
    outputSize,FLAGS_scale_number,FLAGS_scale_gap);
  scaleInputToNetInputs = std::get<0>(tp);
  netInputSizes = std::get<1>(tp);
  scaleInputToOutput = std::get<2>(tp);
  outputResolution = std::get<3>(tp);

  for (int i=0;i<(int)scaleInputToNetInputs.size();++i)
  {
      std::cout << scaleInputToNetInputs[i] << std::endl;
  }

  PoseModel poseModel = PoseModel::BODY_25;
  const string modelFolder = "./";
  const std::vector<HeatMapType>& heatMapTypes = {};
  const ScaleMode heatMapScale = ScaleMode::ZeroToOne;
  const bool addPartCandidates = false;
  const bool enableGoogleLogging = false;
  const int gpuId = -1;

  const auto netInputArray = CvMatToOpInput_createArray(inputImage, scaleInputToNetInputs, netInputSizes);
  auto outputArray = CvMatToOpOutput_createArray(inputImage, scaleInputToOutput, outputResolution);
  //net start
  initUpImpl(poseModel,modelFolder,gpuId,heatMapTypes,
    heatMapScale,addPartCandidates,enableGoogleLogging);
  PoseExtractorCaffe_netInitializationOnThread();
  Array<float> mPoseKeypoints;
  PoseExtractorCaffe_forwardPass(netInputArray, imageSize, 
    scaleInputToNetInputs,mPoseKeypoints);
  renderPose(outputArray, mPoseKeypoints, scaleInputToOutput,-1.f);
  auto outputImage = formatToCvMat(outputArray);


  /*
  std::vector<Prediction> predictions = classifier.Classify(img);

  //Print the top N predictions. 
  for (size_t i = 0; i < predictions.size(); ++i) {
    Prediction p = predictions[i];
    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              << p.first << "\"" << std::endl;
  }
  */
  //Point<int> mWindowedSize=Point<int>{640, 480};
  #ifdef UNIX_GUI
  std::string mWindowName = "ans";
  cv::namedWindow( mWindowName, CV_WINDOW_AUTOSIZE );
  cv::imshow(mWindowName, outputImage);
  cv::waitKey(-1);
  #endif

}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
