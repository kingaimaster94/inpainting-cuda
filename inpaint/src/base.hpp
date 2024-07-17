#include <vector>
#include <cmath>
#include <optional>

class InvalidRange {
public:
    float min;
    float max;
    float value;
    const char* name;
};

class SizeMismatch {
public:
    std::pair<uint32_t, uint32_t> input;
    std::pair<uint32_t, uint32_t> output;
};

class Error {
public:
    enum class ErrorType {
        Image,
        InvalidRange,
        SizeMismatch,
        ExampleGuideMismatch,
        Io,
        UnsupportedOutputFormat,
        NoExamples,
        MapsCountMismatch
    };

    ErrorType type;
    std::string message;

    // Constructors for different error types are responsibility of the user
};

// Dims structure
struct Dims {
    uint32_t width;
    uint32_t height;

    Dims(uint32_t width, uint32_t height) : width(width), height(height) {}
    Dims(uint32_t size) : width(size), height(size) {}
};

// Parameters structure
struct Parameters {
    bool tiling_mode;
    uint32_t nearest_neighbors;
    uint64_t random_sample_locations;
    float cauchy_dispersion;
    float backtrack_percent;
    uint32_t backtrack_stages;
    Dims resize_input;
    Dims output_size;
    float guide_alpha;
    uint64_t random_resolve;
    size_t max_thread_count;
    uint64_t seed;

    Parameters() : tiling_mode(false), nearest_neighbors(50), random_sample_locations(50),
        cauchy_dispersion(1.0f), backtrack_percent(0.5f), backtrack_stages(5),
        resize_input(Dims(0)), output_size(Dims(500)), guide_alpha(0.8f),
        random_resolve(), max_thread_count(), seed(0) {}
};