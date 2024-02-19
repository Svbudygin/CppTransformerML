#ifndef NNLAYERS_H
#define NNLAYERS_H

#include <vector>

class Layer {
public:
    virtual ~Layer() {} // Virtual destructor for polymorphism

    // Forward pass method for the layer
    virtual std::vector<double> forward(const std::vector<double>& input) = 0;
    virtual std::vector<double> backward(const std::vector<double>& gradient) = 0;
};

class LinearLayer : public Layer {
private:
    std::vector<std::vector<double>> weights; // Weights matrix
    std::vector<double> biases; // Biases vector
    std::vector<double> input;
    double learningRate;

public:
    LinearLayer(int inputSize, int outputSize);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& gradient) override;
    void updateWeights(const std::vector<std::vector<double>>& weightGradients,
                       const std::vector<double>& biasGradients,
                       double learningRate);
};

class ReLULayer : public Layer {
private:
    std::vector<double> input;

public:
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& gradient) override;
};

#endif // NNLAYERS_H