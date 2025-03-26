#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sdsl/suffix_array_algorithm.hpp>
#include <sdsl/suffix_arrays.hpp>
#include "./dolma-to-fmindex/library/nlohmann/json.hpp"
#include <unordered_set>
#include <mutex>
#include <vector>

using namespace sdsl;
using namespace std;
using json = nlohmann::json;
using namespace std::chrono;

std::mutex output_mutex; // Mutex to protect shared output JSON

// Function to read file content into a string
std::string read_string_from_file(const std::string &file_path) {
    std::ifstream input_stream(file_path, std::ios_base::binary);
    if (!input_stream) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }
    std::stringstream buffer;
    buffer << input_stream.rdbuf();
    return buffer.str();
}

// Function to parse JSON and extract entities with their names
std::map<std::string, std::vector<std::string>> extract_entities_with_names(const std::string& json_content) {
    std::map<std::string, std::vector<std::string>> entities;
    try {
        json j = json::parse(json_content);
        for (auto& [key, value] : j.items()) {
            if (value.contains("names")) {
                std::vector<std::string> names = value["names"].get<std::vector<std::string>>();
                entities[key] = std::move(names);
            }
        }
    } catch (const json::parse_error& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
    }
    return entities;
}

// Function to process entities using a pre-loaded FM-index
void process_entities(const csa_wt<wt_huff<rrr_vector<127>>, 512, 1024>& fm_index, const std::map<std::string, std::vector<std::string>>& entities, json& output_json) {
    size_t total_entities = entities.size();
    size_t processed_entities = 0;

    // Process each entity and count occurrences
    for (const auto& [entity_id, names] : entities) {
        size_t entity_total_occurrences = 0;

        // Sort names by length in descending order to prioritize longer matches
        std::vector<std::string> sorted_names = names;
        std::sort(sorted_names.begin(), sorted_names.end(), [](const std::string &a, const std::string &b) {
            return b.size() > a.size();
        });

        for (const auto& name : sorted_names) {
            try {
                // Get the count of occurrences using sdsl::count() directly
                size_t count = sdsl::count(fm_index, name.begin(), name.end());
                entity_total_occurrences += count;
            } catch (const std::exception& e) {
                std::cerr << "Error counting entity name '" << name << "' in FM-index: " << e.what() << std::endl;
            }
        }

        // Lock output mutex before updating the shared output JSON
        std::lock_guard<std::mutex> lock(output_mutex);
        output_json[entity_id] = output_json[entity_id].get<size_t>() + entity_total_occurrences;

        // Update progress
        processed_entities++;
        std::cout << "\rProgress: " << processed_entities << "/" << total_entities << " entities processed." << std::flush;
    }
    std::cout << std::endl;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        cout << "Not enough arguments" << endl;
        cout << "Usage: " << argv[0] << " fm_index_file json_file output_file" << endl;
        return 1;
    }

    auto start_time = high_resolution_clock::now();

    string fm_index_file = argv[1];
    string json_file = argv[2];
    string output_file_path = argv[3];

    // Load and parse the JSON file
    string json_content = read_string_from_file(json_file);
    auto entities = extract_entities_with_names(json_content);

    // Load the FM-index once in the main function
    csa_wt<wt_huff<rrr_vector<127>>, 512, 1024> fm_index;
    try {
        std::cout << "Attempting to load FM-index from file: " << fm_index_file << std::endl;
        if (!load_from_file(fm_index, fm_index_file)) {
            throw std::runtime_error("ERROR: Could not load FM-index.");
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception caught while loading FM-index: " << e.what() << std::endl;
        return 1;
    }

    // Create a JSON object for output
    json output_json;
    for (const auto& [entity_id, _] : entities) {
        output_json[entity_id] = 0; // Initialize counts to 0
    }

    // Process entities using the loaded FM-index
    process_entities(fm_index, entities, output_json);

    // Write output JSON to file
    std::ofstream output_file(output_file_path);
    if (output_file.is_open()) {
        output_file << output_json.dump(4); // Pretty print with 4 spaces
        output_file.close();
    } else {
        std::cerr << "Failed to write output JSON to file." << std::endl;
    }

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(end_time - start_time);
    std::cout << "\nExecution Time: " << duration.count() << " seconds." << std::endl;

    return 0;
}
