#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <torch/script.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

struct Args {
    std::string model_path = "../../models/pytorch_simple_cnn_scripted.pt";
    std::string image_dir = "../../prediction_test";
    std::string mapping_path = "../../data/emnist-balanced-mapping.txt";
    std::string result_dir = "../../results";
    int limit = 100;
    double power_watt = 5.0;
};

struct PredictionRecord {
    std::string image_name;
    std::string true_label;
    std::string predicted_label;
    int predicted_class_id;
    double confidence_percent;
    double prediction_time_seconds;
    bool correct;
};

struct TemperatureStats {
    double avg_temperature_c;
    double max_temperature_c;
};

std::string to_lowercase(std::string text) {
    std::transform(
        text.begin(),
        text.end(),
        text.begin(),
        [](unsigned char c) { return std::tolower(c); }
    );
    return text;
}

bool is_image_file(const fs::path& path) {
    std::string ext = to_lowercase(path.extension().string());
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp";
}

Args parse_args(int argc, char* argv[]) {
    Args args;

    for (int i = 1; i < argc; i++) {
        std::string key = argv[i];

        if (key == "--model" && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (key == "--image-dir" && i + 1 < argc) {
            args.image_dir = argv[++i];
        } else if (key == "--mapping" && i + 1 < argc) {
            args.mapping_path = argv[++i];
        } else if (key == "--result-dir" && i + 1 < argc) {
            args.result_dir = argv[++i];
        } else if (key == "--limit" && i + 1 < argc) {
            args.limit = std::stoi(argv[++i]);
        } else if (key == "--power-watt" && i + 1 < argc) {
            args.power_watt = std::stod(argv[++i]);
        }
    }

    return args;
}

double get_file_size_mb(const std::string& path) {
    if (!fs::exists(path)) {
        return 0.0;
    }

    double bytes = static_cast<double>(fs::file_size(path));
    return bytes / (1024.0 * 1024.0);
}

double get_process_memory_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(
        GetCurrentProcess(),
        reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
        sizeof(pmc)
    );

    return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
#else
    std::ifstream statm("/proc/self/statm");

    long size_pages = 0;
    long resident_pages = 0;

    statm >> size_pages >> resident_pages;

    long page_size = sysconf(_SC_PAGESIZE);
    double resident_bytes = static_cast<double>(resident_pages) * page_size;

    return resident_bytes / (1024.0 * 1024.0);
#endif
}

double get_system_memory_percent() {
#ifdef _WIN32
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    GlobalMemoryStatusEx(&mem_status);

    return static_cast<double>(mem_status.dwMemoryLoad);
#else
    std::ifstream meminfo("/proc/meminfo");

    std::string key;
    double value;
    std::string unit;

    double mem_total = 0.0;
    double mem_available = 0.0;

    while (meminfo >> key >> value >> unit) {
        if (key == "MemTotal:") {
            mem_total = value;
        } else if (key == "MemAvailable:") {
            mem_available = value;
        }
    }

    if (mem_total <= 0.0) {
        return -1.0;
    }

    double used = mem_total - mem_available;
    return (used / mem_total) * 100.0;
#endif
}

double get_cpu_temperature_c() {
#ifdef _WIN32
    return -1.0;
#else
    std::ifstream temp_file("/sys/class/thermal/thermal_zone0/temp");

    if (!temp_file.is_open()) {
        return -1.0;
    }

    double temp_milli_c;
    temp_file >> temp_milli_c;

    return temp_milli_c / 1000.0;
#endif
}

void record_temperature(std::vector<double>& temperatures) {
    double temp = get_cpu_temperature_c();

    if (temp >= 0.0) {
        temperatures.push_back(temp);
    }
}

TemperatureStats calculate_temperature_stats(const std::vector<double>& temperatures) {
    if (temperatures.empty()) {
        return {-1.0, -1.0};
    }

    double sum = std::accumulate(temperatures.begin(), temperatures.end(), 0.0);
    double avg = sum / static_cast<double>(temperatures.size());
    double max_temp = *std::max_element(temperatures.begin(), temperatures.end());

    return {avg, max_temp};
}

double calculate_temperature_cost(
    double temperature_c,
    double baseline_temp = 40.0,
    double temp_factor = 0.01
) {
    if (temperature_c < 0.0) {
        return 0.0;
    }

    double extra_temp = std::max(0.0, temperature_c - baseline_temp);
    return extra_temp * temp_factor;
}

double calculate_gcf(
    double power_watt,
    double time_seconds,
    double memory_percent,
    double temperature_c,
    double carbon_factor = 0.000417,
    double memory_factor = 0.001
) {
    double time_hours = time_seconds / 3600.0;

    double energy_carbon_cost = power_watt * time_hours * carbon_factor;
    double memory_cost = memory_percent * memory_factor;
    double temperature_cost = calculate_temperature_cost(temperature_c);

    return energy_carbon_cost + memory_cost + temperature_cost;
}

std::string value_or_na(double value) {
    if (value < 0.0 || std::isnan(value)) {
        return "NA";
    }

    std::ostringstream oss;
    oss << value;
    return oss.str();
}

std::map<int, std::string> load_mapping(const std::string& mapping_path) {
    std::map<int, std::string> label_to_char;

    std::ifstream file(mapping_path);

    if (!file.is_open()) {
        throw std::runtime_error("Mapping file not found: " + mapping_path);
    }

    int label;
    int ascii_code;

    while (file >> label >> ascii_code) {
        char character = static_cast<char>(ascii_code);
        label_to_char[label] = std::string(1, character);
    }

    return label_to_char;
}

std::string get_true_label_from_filename(const std::string& image_path) {
    fs::path path(image_path);
    std::string stem = path.stem().string();

    std::vector<std::string> parts;
    std::stringstream ss(stem);
    std::string item;

    while (std::getline(ss, item, '_')) {
        parts.push_back(item);
    }

    if (parts.size() < 2) {
        throw std::runtime_error(
            "Invalid filename format: " + path.filename().string() +
            ". Use upper_A_01.png, lower_a_01.png, or digit_5_01.png"
        );
    }

    std::string label_type = parts[0];
    std::string label_value = parts[1];

    if (label_type == "upper") {
        std::transform(
            label_value.begin(),
            label_value.end(),
            label_value.begin(),
            [](unsigned char c) { return std::toupper(c); }
        );
        return label_value;
    }

    if (label_type == "lower") {
        std::transform(
            label_value.begin(),
            label_value.end(),
            label_value.begin(),
            [](unsigned char c) { return std::tolower(c); }
        );
        return label_value;
    }

    if (label_type == "digit") {
        return label_value;
    }

    throw std::runtime_error(
        "Invalid label type in filename: " + path.filename().string() +
        ". Use prefix upper, lower, or digit."
    );
}

std::vector<std::string> get_image_files(const std::string& image_dir, int limit) {
    std::vector<std::string> files;

    if (!fs::exists(image_dir)) {
        throw std::runtime_error("Image directory not found: " + image_dir);
    }

    for (const auto& entry : fs::directory_iterator(image_dir)) {
        if (entry.is_regular_file() && is_image_file(entry.path())) {
            files.push_back(entry.path().string());
        }
    }

    std::sort(files.begin(), files.end());

    if (limit > 0 && static_cast<int>(files.size()) > limit) {
        files.resize(limit);
    }

    return files;
}

torch::Tensor preprocess_image(const std::string& image_path) {
    cv::Mat gray = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    if (gray.empty()) {
        throw std::runtime_error("Could not read image: " + image_path);
    }

    cv::Scalar mean_value = cv::mean(gray);

    // If background is white, invert image.
    // After inversion: character = white, background = black.
    if (mean_value[0] > 127.0) {
        cv::bitwise_not(gray, gray);
    }

    cv::Mat binary;
    cv::threshold(gray, binary, 50, 255, cv::THRESH_BINARY);

    std::vector<cv::Point> non_zero_points;
    cv::findNonZero(binary, non_zero_points);

    if (non_zero_points.empty()) {
        throw std::runtime_error("No visible character found in image: " + image_path);
    }

    cv::Rect bounding_box = cv::boundingRect(non_zero_points);
    cv::Mat cropped = binary(bounding_box);

    int original_width = cropped.cols;
    int original_height = cropped.rows;

    double scale = std::min(20.0 / original_width, 20.0 / original_height);

    int new_width = std::max(1, static_cast<int>(std::round(original_width * scale)));
    int new_height = std::max(1, static_cast<int>(std::round(original_height * scale)));

    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_AREA);

    cv::Mat canvas = cv::Mat::zeros(28, 28, CV_8UC1);

    int left = (28 - new_width) / 2;
    int top = (28 - new_height) / 2;

    resized.copyTo(canvas(cv::Rect(left, top, new_width, new_height)));

    torch::Tensor input_tensor = torch::from_blob(
        canvas.data,
        {1, 1, 28, 28},
        torch::kUInt8
    );

    input_tensor = input_tensor.to(torch::kFloat32).div(255.0).clone();

    return input_tensor;
}

void save_deployment_metrics(
    const std::string& result_dir,
    const std::string& model_path,
    double model_file_size_mb,
    double loading_time_seconds,
    double memory_before_mb,
    double memory_after_mb,
    double memory_before_percent,
    double memory_after_percent,
    double cpu_percent,
    double avg_temperature_c,
    double max_temperature_c
) {
    fs::create_directories(result_dir);

    std::string output_path = result_dir + "/libtorch_deployment_metrics.csv";

    std::ofstream file(output_path);

    file << "framework,model_path,model_file_size_mb,model_loading_time_seconds,"
         << "memory_before_loading_mb,memory_after_loading_mb,memory_increase_mb,"
         << "memory_before_percent,memory_after_percent,avg_memory_percent,"
         << "estimated_cpu_percent,avg_temperature_c,max_temperature_c\n";

    double memory_increase_mb = memory_after_mb - memory_before_mb;
    double avg_memory_percent = (memory_before_percent + memory_after_percent) / 2.0;

    file << "LibTorch,"
         << model_path << ","
         << model_file_size_mb << ","
         << loading_time_seconds << ","
         << memory_before_mb << ","
         << memory_after_mb << ","
         << memory_increase_mb << ","
         << value_or_na(memory_before_percent) << ","
         << value_or_na(memory_after_percent) << ","
         << value_or_na(avg_memory_percent) << ","
         << value_or_na(cpu_percent) << ","
         << value_or_na(avg_temperature_c) << ","
         << value_or_na(max_temperature_c) << "\n";

    file.close();

    std::cout << "Saved deployment metrics: " << output_path << std::endl;
}

void save_prediction_results(
    const std::string& result_dir,
    const std::vector<PredictionRecord>& records,
    int total,
    int correct,
    double total_prediction_time,
    double average_latency,
    double throughput,
    double avg_cpu_percent,
    double avg_memory_percent,
    double avg_temperature_c,
    double max_temperature_c,
    double power_watt,
    double gcf_score
) {
    fs::create_directories(result_dir);

    std::string summary_path = result_dir + "/libtorch_prediction_summary.csv";
    std::string details_path = result_dir + "/libtorch_prediction_details.csv";

    double accuracy = total > 0 ? static_cast<double>(correct) / total : 0.0;

    std::ofstream summary(summary_path);

    summary << "framework,phase,total_images,correct_predictions,prediction_accuracy,"
            << "total_prediction_time_seconds,average_latency_seconds_per_image,"
            << "throughput_images_per_second,estimated_cpu_percent,avg_memory_percent,"
            << "avg_temperature_c,max_temperature_c,estimated_power_watt,prediction_gcf_score\n";

    summary << "LibTorch,"
            << "post_deployment_prediction,"
            << total << ","
            << correct << ","
            << accuracy << ","
            << total_prediction_time << ","
            << average_latency << ","
            << throughput << ","
            << value_or_na(avg_cpu_percent) << ","
            << value_or_na(avg_memory_percent) << ","
            << value_or_na(avg_temperature_c) << ","
            << value_or_na(max_temperature_c) << ","
            << power_watt << ","
            << gcf_score << "\n";

    summary.close();

    std::ofstream details(details_path);

    details << "image,true_label,predicted_label,predicted_class_id,"
            << "confidence_percent,prediction_time_seconds,correct\n";

    for (const auto& r : records) {
        details << r.image_name << ","
                << r.true_label << ","
                << r.predicted_label << ","
                << r.predicted_class_id << ","
                << r.confidence_percent << ","
                << r.prediction_time_seconds << ","
                << (r.correct ? "true" : "false") << "\n";
    }

    details.close();

    std::cout << "Saved prediction summary: " << summary_path << std::endl;
    std::cout << "Saved prediction details: " << details_path << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        Args args = parse_args(argc, argv);

        std::cout << "LibTorch Benchmark Started" << std::endl;
        std::cout << "Model path: " << args.model_path << std::endl;
        std::cout << "Image directory: " << args.image_dir << std::endl;
        std::cout << "Mapping path: " << args.mapping_path << std::endl;
        std::cout << "Result directory: " << args.result_dir << std::endl;
        std::cout << "Limit: " << args.limit << std::endl;

        auto label_to_char = load_mapping(args.mapping_path);

        double model_file_size_mb = get_file_size_mb(args.model_path);

        double memory_before_loading_mb = get_process_memory_mb();
        double memory_before_loading_percent = get_system_memory_percent();

        std::vector<double> deployment_temperatures;
        record_temperature(deployment_temperatures);

        auto load_wall_start = std::chrono::high_resolution_clock::now();
        std::clock_t load_cpu_start = std::clock();

        torch::jit::script::Module module = torch::jit::load(args.model_path);
        module.eval();

        // Warm-up inference
        torch::Tensor dummy_input = torch::randn({1, 1, 28, 28});
        {
            torch::NoGradGuard no_grad;
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(dummy_input);
            module.forward(inputs).toTensor();
        }

        std::clock_t load_cpu_end = std::clock();
        auto load_wall_end = std::chrono::high_resolution_clock::now();

        record_temperature(deployment_temperatures);

        TemperatureStats deployment_temp_stats =
            calculate_temperature_stats(deployment_temperatures);

        double loading_time_seconds =
            std::chrono::duration<double>(load_wall_end - load_wall_start).count();

        double loading_cpu_seconds =
            static_cast<double>(load_cpu_end - load_cpu_start) / CLOCKS_PER_SEC;

        double cpu_count = static_cast<double>(
            std::max(1u, std::thread::hardware_concurrency())
        );

        double deployment_cpu_percent =
            loading_time_seconds > 0.0
                ? (loading_cpu_seconds / loading_time_seconds) / cpu_count * 100.0
                : 0.0;

        double memory_after_loading_mb = get_process_memory_mb();
        double memory_after_loading_percent = get_system_memory_percent();

        save_deployment_metrics(
            args.result_dir,
            args.model_path,
            model_file_size_mb,
            loading_time_seconds,
            memory_before_loading_mb,
            memory_after_loading_mb,
            memory_before_loading_percent,
            memory_after_loading_percent,
            deployment_cpu_percent,
            deployment_temp_stats.avg_temperature_c,
            deployment_temp_stats.max_temperature_c
        );

        std::vector<std::string> image_files = get_image_files(args.image_dir, args.limit);

        if (image_files.empty()) {
            throw std::runtime_error("No image files found in: " + args.image_dir);
        }

        std::cout << "Total images found: " << image_files.size() << std::endl;
        std::cout << "Starting LibTorch prediction benchmark..." << std::endl;

        std::vector<PredictionRecord> records;

        int correct = 0;
        int total = 0;

        double memory_before_prediction_percent = get_system_memory_percent();

        std::vector<double> prediction_temperatures;
        record_temperature(prediction_temperatures);

        auto pred_wall_start = std::chrono::high_resolution_clock::now();
        std::clock_t pred_cpu_start = std::clock();

        {
            torch::NoGradGuard no_grad;

            for (const auto& image_path : image_files) {
                fs::path path(image_path);

                std::string true_label = get_true_label_from_filename(image_path);

                torch::Tensor input_tensor = preprocess_image(image_path);

                auto single_start = std::chrono::high_resolution_clock::now();

                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);

                torch::Tensor output = module.forward(inputs).toTensor();
                torch::Tensor probabilities = torch::softmax(output, 1);

                auto max_result = probabilities.max(1);
                torch::Tensor confidence_tensor = std::get<0>(max_result);
                torch::Tensor class_tensor = std::get<1>(max_result);

                int predicted_class_id = class_tensor.item<int>();
                double confidence_percent = confidence_tensor.item<float>() * 100.0;

                auto single_end = std::chrono::high_resolution_clock::now();

                double single_prediction_time =
                    std::chrono::duration<double>(single_end - single_start).count();

                std::string predicted_label = std::to_string(predicted_class_id);

                if (label_to_char.find(predicted_class_id) != label_to_char.end()) {
                    predicted_label = label_to_char[predicted_class_id];
                }

                bool is_correct = predicted_label == true_label;

                if (is_correct) {
                    correct++;
                }

                total++;

                records.push_back({
                    path.filename().string(),
                    true_label,
                    predicted_label,
                    predicted_class_id,
                    confidence_percent,
                    single_prediction_time,
                    is_correct
                });

                record_temperature(prediction_temperatures);
            }
        }

        std::clock_t pred_cpu_end = std::clock();
        auto pred_wall_end = std::chrono::high_resolution_clock::now();

        record_temperature(prediction_temperatures);

        TemperatureStats prediction_temp_stats =
            calculate_temperature_stats(prediction_temperatures);

        double total_prediction_time =
            std::chrono::duration<double>(pred_wall_end - pred_wall_start).count();

        double prediction_cpu_seconds =
            static_cast<double>(pred_cpu_end - pred_cpu_start) / CLOCKS_PER_SEC;

        double prediction_cpu_percent =
            total_prediction_time > 0.0
                ? (prediction_cpu_seconds / total_prediction_time) / cpu_count * 100.0
                : 0.0;

        double memory_after_prediction_percent = get_system_memory_percent();

        double avg_memory_percent =
            (memory_before_prediction_percent + memory_after_prediction_percent) / 2.0;

        double average_latency =
            total > 0 ? total_prediction_time / static_cast<double>(total) : 0.0;

        double throughput =
            total_prediction_time > 0.0
                ? static_cast<double>(total) / total_prediction_time
                : 0.0;

        double gcf_score = calculate_gcf(
            args.power_watt,
            total_prediction_time,
            avg_memory_percent,
            prediction_temp_stats.avg_temperature_c
        );

        save_prediction_results(
            args.result_dir,
            records,
            total,
            correct,
            total_prediction_time,
            average_latency,
            throughput,
            prediction_cpu_percent,
            avg_memory_percent,
            prediction_temp_stats.avg_temperature_c,
            prediction_temp_stats.max_temperature_c,
            args.power_watt,
            gcf_score
        );

        std::cout << "\nLibTorch Prediction Summary" << std::endl;
        std::cout << "Total images: " << total << std::endl;
        std::cout << "Correct predictions: " << correct << std::endl;
        std::cout << "Accuracy: " << static_cast<double>(correct) / total << std::endl;
        std::cout << "Total prediction time: " << total_prediction_time << " seconds" << std::endl;
        std::cout << "Average latency: " << average_latency << " seconds/image" << std::endl;
        std::cout << "Throughput: " << throughput << " images/second" << std::endl;
        std::cout << "Estimated CPU percent: " << prediction_cpu_percent << std::endl;
        std::cout << "Average memory percent: " << avg_memory_percent << std::endl;
        std::cout << "Average temperature: "
                  << value_or_na(prediction_temp_stats.avg_temperature_c) << std::endl;
        std::cout << "Maximum temperature: "
                  << value_or_na(prediction_temp_stats.max_temperature_c) << std::endl;
        std::cout << "Prediction GCF score: " << gcf_score << std::endl;

        return 0;

    } catch (const c10::Error& e) {
        std::cerr << "LibTorch error: " << e.what() << std::endl;
        return 1;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}