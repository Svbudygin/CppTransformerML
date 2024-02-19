#include "nnlayers.h"
#include <cmath> // For ReLU function
#include <cstddef>
#include <stddef.h> 

// Constructor for LinearLayer
LinearLayer::LinearLayer(int inputSize, int outputSize) {
    // Initialize weights and biases with random values or zeros
    // For simplicity, let's initialize weights to 0 and biases to 0
    weights.resize(outputSize, std::vector<double>(inputSize, 0.0));
    biases.resize(outputSize, 0.0);
}

// Forward pass method for LinearLayer
std::vector<double> LinearLayer::forward(const std::vector<double>& input) {
    // Store input for later use in backward pass
    this->input = input;

    // Perform linear transformation: output = weights * input + biases
    std::vector<double> output(weights.size(), 0.0);
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < input.size(); ++j) {
            output[i] += weights[i][j] * input[j];
        }
        output[i] += biases[i];
    }

    return output;
}

void LinearLayer::updateWeights()

// Backward pass method for LinearLayer
std::vector<double> LinearLayer::backward(const std::vector<double>& gradient) {
    // Compute gradients w.r.t. weights and biases
    std::vector<std::vector<double>> weightGradients(weights.size(), std::vector<double>(input.size(), 0.0));
    #include <cstddef> // Add missing include directive for 'size_t'
    
    // weightGradients = std::vector<std::vector<double>>(weights.size(), std::vector<double>(input.size(), 0.0));
    
    // Calculate gradients
    std::vector<double> biasGradients(biases.size(), 0.0);
    size_t i; // Declare 'i' before the loop
    for (i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < input.size(); ++j) {
            weightGradients[i][j] = gradient[i] * input[j];
        }
        biasGradients[i] = gradient[i];
    }

    // Update input gradient
    std::vector<double> inputGradient(input.size(), 0.0);
    for (size_t j = 0; j < input.size(); ++j) {
        for (size_t i = 0; i < weights.size(); ++i) {
            inputGradient[j] += gradient[i] * weights[i][j];
        }
    }

    // Update weights and biases
    updateWeights(weightGradients, biasGradients, learningRate);

    return inputGradient;
}

// Update weights and biases based on gradients and learning rate
void LinearLayer::updateWeights(const std::vector<std::vector<double>>& weightGradients,
                                const std::vector<double>& biasGradients,
                                double learningRate) {
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            weights[i][j] -= learningRate * weightGradients[i][j];
        }
        biases[i] -= learningRate * biasGradients[i];
    }
}

// Forward pass method for ReLULayer
std::vector<double> ReLULayer::forward(const std::vector<double>& input) {
    // Store input for later use in backward pass
    this->input = input;

    // Apply ReLU activation function
    std::vector<double> output(input.size(), 0.0);
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::max(0.0, input[i]);
    }

    return output;
}

// Backward pass method for ReLULayer
std::vector<double> ReLULayer::backward(const std::vector<double>& gradient) {
    // Compute gradient of ReLU activation function
    std::vector<double> inputGradient(input.size(), 0.0);
    for (size_t i = 0; i < input.size(); ++i) {
        inputGradient[i] = (input[i] > 0) ? gradient[i] : 0.0;
    }

    return inputGradient;
}
