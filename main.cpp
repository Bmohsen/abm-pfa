#define ASIO_STANDALONE 

#include "crow_all.h" 
#include "fasttext/src/fasttext.h"
#include <sstream>
#include <iostream>
#include <vector>


std::string vector_to_json(const std::vector<float>& vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i] << (i < vec.size() - 1 ? "," : "");
    }
    oss << "]";
    return oss.str();
}

int main() {
    std::cout << "[ABM Tech Present: NLP For Products Matching] \n";
    // 1. Load the FastText Model
    std::cout << "Loading AI Model..." << std::endl;
    fasttext::FastText ft;
    try {
        ft.loadModel("osayebana_materials_model.bin");
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Model Loaded. Starting Server on Port 8080..." << std::endl;

    crow::SimpleApp app;

    // --- ENDPOINT: Vectorize (For Deduplication) ---
    CROW_ROUTE(app, "/vectorize").methods(crow::HTTPMethod::Post)
    ([&ft](const crow::request& req) {
        auto x = crow::json::load(req.body);
        if (!x) return crow::response(400, "Invalid JSON");
        
        std::string text = x["text"].s();
        
        // FastText: Get Sentence Vector
        // This averages the vectors of all words in the string
        fasttext::Vector vec(ft.getDimension());
        std::istringstream iss(text);
        ft.getSentenceVector(iss, vec);

        // Convert to std::vector<float> for JSON output
        std::vector<float> output_vec;
        for (int i = 0; i < vec.size(); i++) {
            output_vec.push_back(vec[i]);
        }

        crow::json::wvalue result;
        result["vector"] = output_vec; // Returns [0.123, 0.456, ...]
        return crow::response(result);
    });

    // --- ENDPOINT: Predict Category (Supervised) ---
    // If you trained your model with labels (e.g., __label__cement), this works.
    CROW_ROUTE(app, "/predict").methods(crow::HTTPMethod::Post)
    ([&ft](const crow::request& req) {
        auto x = crow::json::load(req.body);
        std::string text = x["text"].s();

        std::vector<std::pair<fasttext::real, std::string>> predictions;
        std::istringstream iss(text);
        
        // Predict top 3 categories
        ft.predictLine(iss, predictions, 3, 0.0); 

        crow::json::wvalue result;
        for (int i = 0; i < predictions.size(); i++) {
             // Remove "__label__" prefix if present
             std::string label = predictions[i].second;
             result["predictions"][i]["label"] = label;
             result["predictions"][i]["score"] = predictions[i].first;
        }
        
        return crow::response(result);
    });

    app.port(8080).multithreaded().run();
}