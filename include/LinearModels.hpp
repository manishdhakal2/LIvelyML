#ifndef LINEAR_MODELS_HPP
#define LINEAR_MODELS_HPP

#include<vector>

class LinearRegression{
    public:
        float lr; // The Learning Rate
        int epoch; // No. of epochs
        
        std::vector<double> y,w,b; // Y- Target variable , W- Weight matrix, B- Bias Matrix
        std::vector<std::vector<double>> x;// The feature space

        LinearRegression(float lr,int epochs); // Constructor

        void init_weights(); // Parameter Initialization

        std::vector <double> dotProduct(std::vector<std::vector<double>> x,std::vector<double> y); // 2d*1d Multiplication

        void fit(std::vector<std::vector<double>> x,std::vector <double> y); // Training the model

        std::vector <double> predict(std::vector<std::vector<double>> x); // Forward Method


};

#endif