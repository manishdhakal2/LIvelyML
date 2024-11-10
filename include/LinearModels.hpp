#ifndef LINEAR_MODELS_HPP
#define LINEAR_MODELS_HPP

#include<vector>

class LinearRegression{
    public:
        double lr; // The Learning Rate
        int epoch; // No. of epochs
        
        std::vector<double> y,w;
        double b; // Y- Target variable , W- Weight matrix, B- Bias 
        std::vector<std::vector<double>> x;// The feature space

        LinearRegression(double lr,int epochs); // Constructor

        void init_weights(); // Parameter Initialization

        double MSE(std::vector<double> y,std::vector <double> y_pred);

        std::vector <double> dotProduct(std::vector<std::vector<double>> x,std::vector<double> y); // 2d*1d Multiplication

        void fit(std::vector<std::vector<double>> x,std::vector <double> y); // Training the model

        std::vector <double> predict(std::vector<std::vector<double>> x); // Forward Method


};

#endif