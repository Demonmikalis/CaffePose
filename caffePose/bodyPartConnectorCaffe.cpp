#include "bodyPartConnectorCaffe.hpp"
#ifdef USE_CAFFE
    #include <caffe/blob.hpp>
#endif

namespace op
{

    //TODO:MR.Black Define some body constants
    const bool COCO_CHALLENGE = false;
    const auto POSE_MAX_PEOPLE = 127u;
    // Constant Array Parameters
    // POSE_NUMBER_BODY_PARTS equivalent to size of std::map POSE_BODY_XX_BODY_PARTS - 1 (removing background)
    const std::array<unsigned int, (int)PoseModel::Size> POSE_NUMBER_BODY_PARTS{
        25, 18, 15, 15, 19, 19, 59, 19, 19, 25, 65, 12
    };
    const std::array<std::vector<unsigned int>, (int)PoseModel::Size> POSE_BODY_PART_PAIRS{
        // BODY_25
        std::vector<unsigned int>{
            1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18,   14,19,19,20,14,21, 11,22,22,23,11,24
        },
        // COCO
        std::vector<unsigned int>{
            1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10,  1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17,  2,16,  5,17
        },
        // MPI_15
        std::vector<unsigned int>{},
        // MPI_15_4
        std::vector<unsigned int>{},
        // BODY_19
        std::vector<unsigned int>{
            1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18
        },
        // BODY_19_X2
        std::vector<unsigned int>{
            1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18
        },
        // BODY_59
        std::vector<unsigned int>{
            1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18,// Body
            7,19, 19,20, 20,21, 21,22, 7,23, 23,24, 24,25, 25,26, 7,27, 27,28, 28,29, 29,30, 7,31, 31,32, 32,33, 33,34, 7,35, 35,36, 36,37, 37,38,      // Left hand
            4,39, 39,40, 40,41, 41,42, 4,43, 43,44, 44,45, 45,46, 4,47, 47,48, 48,49, 49,50, 4,51, 51,52, 52,53, 53,54, 4,55, 55,56, 56,57, 57,58       // Right hand
        },
        // BODY_19N
        std::vector<unsigned int>{
            1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18
        },
        // BODY_19b
        std::vector<unsigned int>{
            1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18, 2,9, 5,12
        },
        // BODY_25_19
        std::vector<unsigned int>{
            1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18,   14,19,19,20,14,21, 11,22,22,23,11,24
        },
        // BODY_65
        std::vector<unsigned int>{
            1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18,   14,19,19,20,14,21, 11,22,22,23,11,24,
            7,25, 25,26, 26,27, 27,28, 7,29, 29,30, 30,31, 31,32, 7,33, 33,34, 34,35, 35,36, 7,37, 37,38, 38,39, 39,40, 7,41, 41,42, 42,43, 43,44,      // Left hand
            4,45, 45,46, 46,47, 47,48, 4,49, 49,50, 50,51, 51,52, 4,53, 53,54, 54,55, 55,56, 4,57, 57,58, 58,59, 59,60, 4,61, 61,62, 62,63, 63,64       // Right hand
        },
        // CAR_12
        std::vector<unsigned int>{
            4,5,   4,6,   4,0,   0,2,   4,8,   8,10,   5,7,   5,1,   1,3,   5,9,   9,11
        },
    };

    const std::array<std::vector<unsigned int>, (int)PoseModel::Size> POSE_MAP_INDEX{
        // BODY_25
        std::vector<unsigned int>{
            0,1, 14,15, 22,23, 16,17, 18,19, 24,25, 26,27, 6,7, 2,3, 4,5, 8,9, 10,11, 12,13, 30,31, 32,33, 36,37, 34,35, 38,39, 20,21, 28,29, 40,41,42,43,44,45, 46,47,48,49,50,51
        },
        // COCO
        std::vector<unsigned int>{
            12,13, 20,21, 14,15, 16,17, 22,23, 24,25, 0,1, 2,3, 4,5, 6,7, 8,9, 10,11, 28,29, 30,31, 34,35, 32,33, 36,37, 18,19, 26,27
        },
        // MPI_15
        std::vector<unsigned int>{
            0,1, 2,3, 4,5, 6,7, 8,9, 10,11, 12,13, 14,15, 16,17, 18,19, 20,21, 22,23, 24,25, 26,27
        },
        // MPI_15_4
        std::vector<unsigned int>{
            0,1, 2,3, 4,5, 6,7, 8,9, 10,11, 12,13, 14,15, 16,17, 18,19, 20,21, 22,23, 24,25, 26,27
        },
        // BODY_19
        std::vector<unsigned int>{
            0,1, 14,15, 22,23, 16,17, 18,19, 24,25, 26,27, 6,7, 2,3, 4,5, 8,9, 10,11, 12,13, 30,31, 32,33, 36,37, 34,35, 38,39, 20,21, 28,29
        },
        // BODY_19_X2
        std::vector<unsigned int>{
            0,1, 14,15, 22,23, 16,17, 18,19, 24,25, 26,27, 6,7, 2,3, 4,5, 8,9, 10,11, 12,13, 30,31, 32,33, 36,37, 34,35, 38,39, 20,21, 28,29
        },
        // BODY_59
        std::vector<unsigned int>{
            0,1, 14,15, 22,23, 16,17, 18,19, 24,25, 26,27, 6,7, 2,3, 4,5, 8,9, 10,11, 12,13, 30,31, 32,33, 36,37, 34,35, 38,39, 20,21, 28,29, // Body
            40,41, 42,43, 44,45, 46,47, 48,49, 14,51, 52,53, 54,55, 56,57, 58,59,
            60,61, 62,63, 64,65, 66,67, 68,69, 70,71, 72,73, 74,75, 76,77, 78,79,// Left hand
            80,81, 82,83, 84,85, 86,87, 88,89, 90,91, 92,93, 94,95, 96,97, 98,99,
            100,101, 102,103, 104,105, 106,107, 108,109, 110,111, 112,113, 114,115, 116,117, 118,119 // Right hand
        },
        // BODY_19N
        std::vector<unsigned int>{
            0,1, 14,15, 22,23, 16,17, 18,19, 24,25, 26,27, 6,7, 2,3, 4,5, 8,9, 10,11, 12,13, 30,31, 32,33, 36,37, 34,35, 38,39, 20,21, 28,29
        },
        // BODY_19b
        std::vector<unsigned int>{
            0,1, 14,15, 22,23, 16,17, 18,19, 24,25, 26,27, 6,7, 2,3, 4,5, 8,9, 10,11, 12,13, 30,31, 32,33, 36,37, 34,35, 38,39, 20,21, 28,29, 60,61, 62,63
        },
        // BODY_25_19
        std::vector<unsigned int>{
            0,1, 14,15, 22,23, 16,17, 18,19, 24,25, 26,27, 6,7, 2,3, 4,5, 8,9, 10,11, 12,13, 30,31, 32,33, 36,37, 34,35, 38,39, 20,21, 28,29, 40,41,42,43,44,45, 46,47,48,49,50,51
        },
        // BODY_65
        std::vector<unsigned int>{
            0,1, 14,15, 22,23, 16,17, 18,19, 24,25, 26,27, 6,7, 2,3, 4,5, 8,9, 10,11, 12,13, 30,31, 32,33, 36,37, 34,35, 38,39, 20,21, 28,29, 40,41,42,43,44,45, 46,47,48,49,50,51, // Body
            52,53, 54,55, 56,57, 58,59, 60,61, 62,63, 64,65, 66,67, 68,69, 70,71,
            72,73, 74,75, 76,77, 78,79, 80,81, 82,83, 84,85, 86,87, 88,89, 90,91,                                                                                                   // Left hand
            92,93, 94,95, 96,97, 98,99, 100,101, 102,103, 104,105, 106,107, 108,109, 110,111,
            112,113, 114,115, 116,117, 118,119, 120,121, 122,123, 124,125, 126,127, 128,129, 130,131                                                                                // Right hand
        },
        // CAR_12
        std::vector<unsigned int>{
            0,1,   2,3,   4,5,   6,7,   8,9,  10,11,  12,13, 14,15, 16,17, 18,19, 20,21
        },
    };

    unsigned int getPoseNumberBodyParts(const PoseModel poseModel)
    {
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

    const std::vector<unsigned int>& getPosePartPairs(const PoseModel poseModel)
    {
        try
        {
            return POSE_BODY_PART_PAIRS.at((int)poseModel);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return POSE_BODY_PART_PAIRS[(int)poseModel];
        }
    }

    const std::vector<unsigned int>& getPoseMapIndex(const PoseModel poseModel)
    {
        try
        {
            return POSE_MAP_INDEX.at((int)poseModel);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return POSE_MAP_INDEX[(int)poseModel];
        }
    }

    template <typename T>
    void connectBodyPartsCpu(Array<T>& poseKeypoints, Array<T>& poseScores, const T* const heatMapPtr,
                             const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
                             const int maxPeaks, const T interMinAboveThreshold, const T interThreshold,
                             const int minSubsetCnt, const T minSubsetScore, const T scaleFactor)
    {
        try
        {
            // Parts Connection
            const auto& bodyPartPairs = getPosePartPairs(poseModel);
            const auto& mapIdx = getPoseMapIndex(poseModel);
            const auto numberBodyParts = getPoseNumberBodyParts(poseModel);
            const auto numberBodyPartsAndBkg = numberBodyParts + 1;
            const auto numberBodyPartPairs = bodyPartPairs.size() / 2;
            if (numberBodyParts == 0)
                error("Invalid value of numberBodyParts, it must be positive, not " + to_string(numberBodyParts),
                      __LINE__, __FUNCTION__, __FILE__);

            // Vector<int> = Each body part + body parts counter; double = subsetScore
            std::vector<std::pair<std::vector<int>, double>> subset;
            const auto subsetCounterIndex = numberBodyParts;
            const auto subsetSize = numberBodyParts+1;

            const auto peaksOffset = 3*(maxPeaks+1);
            const auto heatMapOffset = heatMapSize.area();

            for (auto pairIndex = 0u; pairIndex < numberBodyPartPairs; pairIndex++)
            {
                const auto bodyPartA = bodyPartPairs[2*pairIndex];
                const auto bodyPartB = bodyPartPairs[2*pairIndex+1];
                const auto* candidateAPtr = peaksPtr + bodyPartA*peaksOffset;
                const auto* candidateBPtr = peaksPtr + bodyPartB*peaksOffset;
                const auto numberA = intRound(candidateAPtr[0]);
                const auto numberB = intRound(candidateBPtr[0]);

                // Add parts into the subset in special case
                if (numberA == 0 || numberB == 0)
                {
                    // Change w.r.t. other
                    if (numberA == 0) // numberB == 0 or not
                    {
                        if (numberBodyParts != 15)
                        {
                            for (auto i = 1; i <= numberB; i++)
                            {
                                bool num = false;
                                const auto indexB = bodyPartB;
                                for (auto j = 0u; j < subset.size(); j++)
                                {
                                    const auto off = (int)bodyPartB*peaksOffset + i*3 + 2;
                                    if (subset[j].first[indexB] == off)
                                    {
                                        num = true;
                                        break;
                                    }
                                }
                                if (!num)
                                {
                                    std::vector<int> rowVector(subsetSize, 0);
                                    // Store the index
                                    rowVector[ bodyPartB ] = bodyPartB*peaksOffset + i*3 + 2;
                                    // Last number in each row is the parts number of that person
                                    rowVector[subsetCounterIndex] = 1;
                                    const auto subsetScore = candidateBPtr[i*3+2];
                                    // Second last number in each row is the total score
                                    subset.emplace_back(std::make_pair(rowVector, subsetScore));
                                }
                            }
                        }
                        else
                        {
                            for (auto i = 1; i <= numberB; i++)
                            {
                                std::vector<int> rowVector(subsetSize, 0);
                                // Store the index
                                rowVector[ bodyPartB ] = bodyPartB*peaksOffset + i*3 + 2;
                                // Last number in each row is the parts number of that person
                                rowVector[subsetCounterIndex] = 1;
                                // Second last number in each row is the total score
                                const auto subsetScore = candidateBPtr[i*3+2];
                                subset.emplace_back(std::make_pair(rowVector, subsetScore));
                            }
                        }
                    }
                    else // if (numberA != 0 && numberB == 0)
                    {
                        if (numberBodyParts != 15)
                        {
                            for (auto i = 1; i <= numberA; i++)
                            {
                                bool num = false;
                                const auto indexA = bodyPartA;
                                for (auto j = 0u; j < subset.size(); j++)
                                {
                                    const auto off = (int)bodyPartA*peaksOffset + i*3 + 2;
                                    if (subset[j].first[indexA] == off)
                                    {
                                        num = true;
                                        break;
                                    }
                                }
                                if (!num)
                                {
                                    std::vector<int> rowVector(subsetSize, 0);
                                    // Store the index
                                    rowVector[ bodyPartA ] = bodyPartA*peaksOffset + i*3 + 2;
                                    // Last number in each row is the parts number of that person
                                    rowVector[subsetCounterIndex] = 1;
                                    // Second last number in each row is the total score
                                    const auto subsetScore = candidateAPtr[i*3+2];
                                    subset.emplace_back(std::make_pair(rowVector, subsetScore));
                                }
                            }
                        }
                        else
                        {
                            for (auto i = 1; i <= numberA; i++)
                            {
                                std::vector<int> rowVector(subsetSize, 0);
                                // Store the index
                                rowVector[ bodyPartA ] = bodyPartA*peaksOffset + i*3 + 2;
                                // Last number in each row is the parts number of that person
                                rowVector[subsetCounterIndex] = 1;
                                // Second last number in each row is the total score
                                const auto subsetScore = candidateAPtr[i*3+2];
                                subset.emplace_back(std::make_pair(rowVector, subsetScore));
                            }
                        }
                    }
                }
                else // if (numberA != 0 && numberB != 0)
                {
                    std::vector<std::tuple<double, int, int>> temp;
                    const auto* mapX = heatMapPtr + (numberBodyPartsAndBkg + mapIdx[2*pairIndex]) * heatMapOffset;
                    const auto* mapY = heatMapPtr + (numberBodyPartsAndBkg + mapIdx[2*pairIndex+1]) * heatMapOffset;
                    for (auto i = 1; i <= numberA; i++)
                    {
                        for (auto j = 1; j <= numberB; j++)
                        {
                            const auto vectorAToBX = candidateBPtr[j*3] - candidateAPtr[i*3];
                            const auto vectorAToBY = candidateBPtr[j*3+1] - candidateAPtr[i*3+1];
                            const auto vectorAToBMax = fastMax(std::abs(vectorAToBX), std::abs(vectorAToBY));
                            const auto numberPointsInLine = fastMax(5, fastMin(25, intRound(std::sqrt(5*vectorAToBMax))));
                            const auto vectorNorm = T(std::sqrt( vectorAToBX*vectorAToBX + vectorAToBY*vectorAToBY ));
                            // If the peaksPtr are coincident. Don't connect them.
                            if (vectorNorm > 1e-6)
                            {
                                const auto sX = candidateAPtr[i*3];
                                const auto sY = candidateAPtr[i*3+1];
                                const auto vectorAToBNormX = vectorAToBX/vectorNorm;
                                const auto vectorAToBNormY = vectorAToBY/vectorNorm;

                                auto sum = 0.;
                                auto count = 0;
                                const auto vectorAToBXInLine = vectorAToBX/numberPointsInLine;
                                const auto vectorAToBYInLine = vectorAToBY/numberPointsInLine;
                                for (auto lm = 0; lm < numberPointsInLine; lm++)
                                {
                                    const auto mX = fastMin(heatMapSize.x-1, intRound(sX + lm*vectorAToBXInLine));
                                    const auto mY = fastMin(heatMapSize.y-1, intRound(sY + lm*vectorAToBYInLine));
                                    //checkGE(mX, 0, "", __LINE__, __FUNCTION__, __FILE__);
                                    //checkGE(mY, 0, "", __LINE__, __FUNCTION__, __FILE__);
                                    const auto idx = mY * heatMapSize.x + mX;
                                    const auto score = (vectorAToBNormX*mapX[idx] + vectorAToBNormY*mapY[idx]);
                                    if (score > interThreshold)
                                    {
                                        sum += score;
                                        count++;
                                    }
                                }

                                // parts score + connection score
                                if (count/(float)numberPointsInLine > interMinAboveThreshold)
                                    temp.emplace_back(std::make_tuple(sum/count, i, j));
                            }
                        }
                    }

                    // select the top minAB connection, assuming that each part occur only once
                    // sort rows in descending order based on parts + connection score
                    if (!temp.empty())
                        std::sort(temp.begin(), temp.end(), std::greater<std::tuple<T, int, int>>());

                    std::vector<std::tuple<int, int, double>> connectionK;
                    const auto minAB = fastMin(numberA, numberB);
                    std::vector<int> occurA(numberA, 0);
                    std::vector<int> occurB(numberB, 0);
                    auto counter = 0;
                    for (auto row = 0u; row < temp.size(); row++)
                    {
                        const auto score = std::get<0>(temp[row]);
                        const auto x = std::get<1>(temp[row]);
                        const auto y = std::get<2>(temp[row]);
                        if (!occurA[x-1] && !occurB[y-1])
                        {
                            connectionK.emplace_back(std::make_tuple(bodyPartA*peaksOffset + x*3 + 2,
                                                                     bodyPartB*peaksOffset + y*3 + 2,
                                                                     score));
                            counter++;
                            if (counter==minAB)
                                break;
                            occurA[x-1] = 1;
                            occurB[y-1] = 1;
                        }
                    }

                    // Cluster all the body part candidates into subset based on the part connection
                    if (!connectionK.empty())
                    {
                        // initialize first body part connection 15&16
                        if (pairIndex==0)
                        {
                            for (const auto connectionKI : connectionK)
                            {
                                std::vector<int> rowVector(numberBodyParts+3, 0);
                                const auto indexA = std::get<0>(connectionKI);
                                const auto indexB = std::get<1>(connectionKI);
                                const auto score = std::get<2>(connectionKI);
                                rowVector[bodyPartPairs[0]] = indexA;
                                rowVector[bodyPartPairs[1]] = indexB;
                                rowVector[subsetCounterIndex] = 2;
                                // add the score of parts and the connection
                                const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                                subset.emplace_back(std::make_pair(rowVector, subsetScore));
                            }
                        }
                        // Add ears connections (in case person is looking to opposite direction to camera)
                        else if (
                            (numberBodyParts == 18 && (pairIndex==17 || pairIndex==18))
                            || ((numberBodyParts == 19 || numberBodyParts == 25 || numberBodyParts == 59
                                 || numberBodyParts == 65)
                                && (pairIndex==18 || pairIndex==19))
                            || (poseModel == PoseModel::BODY_19b
                                && (pairIndex == numberBodyPartPairs-1 || pairIndex == numberBodyPartPairs-2))
                            )
                        {
                            for (const auto& connectionKI : connectionK)
                            {
                                const auto indexA = std::get<0>(connectionKI);
                                const auto indexB = std::get<1>(connectionKI);
                                for (auto& subsetJ : subset)
                                {
                                    auto& subsetJFirst = subsetJ.first[bodyPartA];
                                    auto& subsetJFirstPlus1 = subsetJ.first[bodyPartB];
                                    if (subsetJFirst == indexA && subsetJFirstPlus1 == 0)
                                        subsetJFirstPlus1 = indexB;
                                    else if (subsetJFirstPlus1 == indexB && subsetJFirst == 0)
                                        subsetJFirst = indexA;
                                }
                            }
                        }
                        else
                        {
                            // A is already in the subset, find its connection B
                            for (const auto& connectionKI : connectionK)
                            {
                                const auto indexA = std::get<0>(connectionKI);
                                const auto indexB = std::get<1>(connectionKI);
                                const auto score = std::get<2>(connectionKI);
                                auto num = 0;
                                for (auto& subsetJ : subset)
                                {
                                    if (subsetJ.first[bodyPartA] == indexA)
                                    {
                                        subsetJ.first[bodyPartB] = indexB;
                                        num++;
                                        subsetJ.first[subsetCounterIndex] = subsetJ.first[subsetCounterIndex] + 1;
                                        subsetJ.second += peaksPtr[indexB] + score;
                                    }
                                }
                                // if can not find partA in the subset, create a new subset
                                if (num==0)
                                {
                                    std::vector<int> rowVector(subsetSize, 0);
                                    rowVector[bodyPartA] = indexA;
                                    rowVector[bodyPartB] = indexB;
                                    rowVector[subsetCounterIndex] = 2;
                                    const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                                    subset.emplace_back(std::make_pair(rowVector, subsetScore));
                                }
                            }
                        }
                    }
                }
            }

            // Delete people below the following thresholds:
                // a) minSubsetCnt: removed if less than minSubsetCnt body parts
                // b) minSubsetScore: removed if global score smaller than this
                // c) POSE_MAX_PEOPLE: keep first POSE_MAX_PEOPLE people above thresholds
            auto numberPeople = 0;
            std::vector<int> validSubsetIndexes;
            validSubsetIndexes.reserve(fastMin((size_t)POSE_MAX_PEOPLE, subset.size()));
            for (auto index = 0u ; index < subset.size() ; index++)
            {
                auto subsetCounter = subset[index].first[subsetCounterIndex];
                // Foot keypoints do not affect subsetCounter (too many false positives,
                // same foot usually appears as both left and right keypoints)
                // Pros: Removed tons of false positives
                // Cons: Standalone leg will never be recorded
                if (!COCO_CHALLENGE && numberBodyParts == 25)
                {
                    // No consider foot keypoints for that
                    for (auto i = 19 ; i < 25 ; i++)
                        subsetCounter -= (subset[index].first.at(i) > 0);
                }
                const auto subsetScore = subset[index].second;
                if (subsetCounter >= minSubsetCnt && (subsetScore/subsetCounter) >= minSubsetScore)
                {
                    numberPeople++;
                    validSubsetIndexes.emplace_back(index);
                    if (numberPeople == POSE_MAX_PEOPLE)
                        break;
                }
                else if ((subsetCounter < 1 && numberBodyParts != 25) || subsetCounter < 0)
                    error("Bad subsetCounter. Bug in this function if this happens.",
                          __LINE__, __FUNCTION__, __FILE__);
            }

            // Fill and return poseKeypoints
            if (numberPeople > 0)
            {
                poseKeypoints.reset({numberPeople, (int)numberBodyParts, 3});
                poseScores.reset(numberPeople);
            }
            else
            {
                poseKeypoints.reset();
                poseScores.reset();
            }
            const auto numberBodyPartsAndPAFs = numberBodyParts + numberBodyPartPairs;
            for (auto person = 0u ; person < validSubsetIndexes.size() ; person++)
            {
                const auto& subsetPair = subset[validSubsetIndexes[person]];
                const auto& subsetI = subsetPair.first;
                for (auto bodyPart = 0u; bodyPart < numberBodyParts; bodyPart++)
                {
                    const auto baseOffset = (person*numberBodyParts + bodyPart) * 3;
                    const auto bodyPartIndex = subsetI[bodyPart];
                    if (bodyPartIndex > 0)
                    {
                        poseKeypoints[baseOffset] = peaksPtr[bodyPartIndex-2] * scaleFactor;
                        poseKeypoints[baseOffset + 1] = peaksPtr[bodyPartIndex-1] * scaleFactor;
                        poseKeypoints[baseOffset + 2] = peaksPtr[bodyPartIndex];
                    }
                    else
                    {
                        poseKeypoints[baseOffset] = 0.f;
                        poseKeypoints[baseOffset + 1] = 0.f;
                        poseKeypoints[baseOffset + 2] = 0.f;
                    }
                }
                poseScores[person] = subsetPair.second / (float)(numberBodyPartsAndPAFs);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    BodyPartConnectorCaffe<T>::BodyPartConnectorCaffe()
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
    void BodyPartConnectorCaffe<T>::Reshape(const std::vector<caffe::Blob<T>*>& bottom)
    {
        try
        {
            #ifdef USE_CAFFE
                auto heatMapsBlob = bottom.at(0);
                auto peaksBlob = bottom.at(1);

                // Top shape
                const auto maxPeaks = peaksBlob->shape(2) - 1;
                const auto numberBodyParts = peaksBlob->shape(1);

                // Array sizes
                mTopSize = std::array<int, 4>{1, maxPeaks, numberBodyParts, 3};
                mHeatMapsSize = std::array<int, 4>{heatMapsBlob->shape(0), heatMapsBlob->shape(1),
                                                   heatMapsBlob->shape(2), heatMapsBlob->shape(3)};
                mPeaksSize = std::array<int, 4>{peaksBlob->shape(0), peaksBlob->shape(1), peaksBlob->shape(2),
                                                peaksBlob->shape(3)};
            #else
                UNUSED(bottom);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::setPoseModel(const PoseModel poseModel)
    {
        try
        {
            mPoseModel = {poseModel};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::setInterMinAboveThreshold(const T interMinAboveThreshold)
    {
        try
        {
            mInterMinAboveThreshold = {interMinAboveThreshold};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::setInterThreshold(const T interThreshold)
    {
        try
        {
            mInterThreshold = {interThreshold};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::setMinSubsetCnt(const int minSubsetCnt)
    {
        try
        {
            mMinSubsetCnt = {minSubsetCnt};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::setMinSubsetScore(const T minSubsetScore)
    {
        try
        {
            mMinSubsetScore = {minSubsetScore};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::setScaleNetToOutput(const T scaleNetToOutput)
    {
        try
        {
            mScaleNetToOutput = {scaleNetToOutput};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::Forward_cpu(const std::vector<caffe::Blob<T>*>& bottom, Array<T>& poseKeypoints,
                                                Array<T>& poseScores)
    {
        try
        {
            #ifdef USE_CAFFE
                const auto heatMapsBlob = bottom.at(0);
                const auto* const heatMapsPtr = heatMapsBlob->cpu_data();                 // ~8.5 ms COCO, 27ms BODY_59
                const auto* const peaksPtr = bottom.at(1)->cpu_data();                    // ~0.02ms
                const auto maxPeaks = mTopSize[1];
                connectBodyPartsCpu(poseKeypoints, poseScores, heatMapsPtr, peaksPtr, mPoseModel,
                                    Point<int>{heatMapsBlob->shape(3), heatMapsBlob->shape(2)},
                                    maxPeaks, mInterMinAboveThreshold, mInterThreshold,
                                    mMinSubsetCnt, mMinSubsetScore, mScaleNetToOutput);
            #else
                UNUSED(bottom);
                UNUSED(poseKeypoints);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void BodyPartConnectorCaffe<T>::Forward_gpu(const std::vector<caffe::Blob<T>*>& bottom, Array<T>& poseKeypoints,
                                                Array<T>& poseScores)
    {
        try
        {
            #if defined USE_CAFFE && defined USE_CUDA
                const auto heatMapsBlob = bottom.at(0);
                const auto* const heatMapsPtr = heatMapsBlob->cpu_data();
                const auto* const peaksPtr = bottom.at(1)->cpu_data();
                const auto* const heatMapsGpuPtr = heatMapsBlob->gpu_data();
                const auto* const peaksGpuPtr = bottom.at(1)->gpu_data();
                const auto maxPeaks = mTopSize[1];
                connectBodyPartsGpu(poseKeypoints, poseScores, heatMapsPtr, peaksPtr, mPoseModel,
                                    Point<int>{heatMapsBlob->shape(3), heatMapsBlob->shape(2)},
                                    maxPeaks, mInterMinAboveThreshold, mInterThreshold,
                                    mMinSubsetCnt, mMinSubsetScore, mScaleNetToOutput,
                                    heatMapsGpuPtr, peaksGpuPtr);
            #else
                UNUSED(bottom);
                UNUSED(poseKeypoints);
                UNUSED(poseScores);
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
    void BodyPartConnectorCaffe<T>::Backward_cpu(const std::vector<caffe::Blob<T>*>& top,
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
    void BodyPartConnectorCaffe<T>::Backward_gpu(const std::vector<caffe::Blob<T>*>& top,
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

    COMPILE_TEMPLATE_FLOATING_TYPES_CLASS(BodyPartConnectorCaffe);
}
