#ifndef OPENPOSE_CORE_ENUM_CLASSES_HPP
#define OPENPOSE_CORE_ENUM_CLASSES_HPP

namespace op
{
    enum class ScaleMode : unsigned char
    {
        InputResolution,
        NetOutputResolution,
        OutputResolution,
        ZeroToOne, // [0, 1]
        PlusMinusOne, // [-1, 1]
        UnsignedChar, // [0, 255]
        NoScale,
    };

    enum class HeatMapType : unsigned char
    {
        Parts,
        Background,
        PAFs,
    };

    enum class RenderMode : unsigned char
    {
        None,
        Cpu,
        Gpu,
    };

    enum class ElementToRender : unsigned char
    {
        Skeleton,
        Background,
        AddKeypoints,
        AddPAFs,
    };

    enum class ErrorMode : unsigned char
    {
        StdRuntimeError,
        FileLogging,
        StdCerr,
        All,
    };

    enum class LogMode : unsigned char
    {
        FileLogging,
        StdCout,
        All,
    };

    enum class Priority : unsigned char
    {
        None = 0,
        Low = 1,
        Normal = 2,
        High = 3,
        Max = 4,
        NoOutput = 255,
    };

    enum class PoseModel : unsigned char
    {
        /**
         * COCO + 6 foot keypoints + neck + lower abs model, with 25+1 components (see poseParameters.hpp for details).
         */
        BODY_25 = 0,
        COCO_18,        /**< COCO model + neck, with 18+1 components (see poseParameters.hpp for details). */
        MPI_15,         /**< MPI model, with 15+1 components (see poseParameters.hpp for details). */
        MPI_15_4,       /**< Variation of the MPI model, reduced number of CNN stages to 4: faster but less accurate.*/
        BODY_19,        /**< Experimental. Do not use. */
        BODY_19_X2,     /**< Experimental. Do not use. */
        BODY_59,        /**< Experimental. Do not use. */
        BODY_19N,       /**< Experimental. Do not use. */
        BODY_19b,       /**< Experimental. Do not use. */
        BODY_25_19,     /**< Experimental. Do not use. */
        BODY_65,        /**< Experimental. Do not use. */
        CAR_12,         /**< Experimental. Do not use. */
        Size,
    };

    enum class PoseProperty : unsigned char
    {
        NMSThreshold = 0,
        ConnectInterMinAboveThreshold,
        ConnectInterThreshold,
        ConnectMinSubsetCnt,
        ConnectMinSubsetScore,
        Size,
    };
}

#endif // OPENPOSE_CORE_ENUM_CLASSES_HPP
