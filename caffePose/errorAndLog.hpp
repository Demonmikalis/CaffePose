#ifndef OPENPOSE_UTILITIES_ERROR_AND_LOG_HPP
#define OPENPOSE_UTILITIES_ERROR_AND_LOG_HPP

#include <atomic>
#include <mutex>
#include <sstream> // std::stringstream
#include <string>
#include <vector>
#include "macros.hpp"
#include "enumClasses.hpp"

namespace op
{
    template<typename T>
    std::string to_string(const T& message)
    {
        // Message -> ostringstream
        std::ostringstream oss;
        oss << message;
        // ostringstream -> std::string
        return oss.str();
    }

    // Error managment - How to use:
        // error(message, __LINE__, __FUNCTION__, __FILE__);
    OP_API void error(const std::string& message, const int line = -1, const std::string& function = "",
                      const std::string& file = "");

    template<typename T>
    inline void error(const T& message, const int line = -1, const std::string& function = "",
                      const std::string& file = "")
    {
        error(to_string(message), line, function, file);
    }

    // Printing info - How to use:
        // It will print info if desiredPriority >= sPriorityThreshold
        // log(message, desiredPriority, __LINE__, __FUNCTION__, __FILE__);
    OP_API void log(const std::string& message, const Priority priority = Priority::Max, const int line = -1,
                    const std::string& function = "", const std::string& file = "");

    template<typename T>
    inline void log(const T& message, const Priority priority = Priority::Max, const int line = -1,
                    const std::string& function = "", const std::string& file = "")
    {
        log(to_string(message), priority, line, function, file);
    }

    // If only desired on debug mode (no computational cost at all on release mode):
        // It will print info if desiredPriority >= sPriorityThreshold
        // dLog(message, desiredPriority, __LINE__, __FUNCTION__, __FILE__);
    template<typename T>
    inline void dLog(const T& message, const Priority priority = Priority::Max, const int line = -1,
                     const std::string& function = "", const std::string& file = "")
    {
        #ifndef NDEBUG
            log(message, priority, line, function, file);
        #else
            UNUSED(message);
            UNUSED(priority);
            UNUSED(line);
            UNUSED(function);
            UNUSED(file);
        #endif
    }

    // This class is thread-safe
    class OP_API ConfigureError
    {
    public:
        static std::vector<ErrorMode> getErrorModes();

        static void setErrorModes(const std::vector<ErrorMode>& errorModes);
    };

    // This class is thread-safe
    class OP_API ConfigureLog
    {
    public:
        static Priority getPriorityThreshold();

        static const std::vector<LogMode>& getLogModes();

        static void setPriorityThreshold(const Priority priorityThreshold);

        static void setLogModes(const std::vector<LogMode>& loggingModes);
    };
}

#endif // OPENPOSE_UTILITIES_ERROR_AND_LOG_HPP
