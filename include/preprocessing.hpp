#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include<vector>

class StandardScaler{

    private:
        std::vector<double> mean,stdev; //Vectors for mean and stddev
        
    public:
        void fit(std::vector<std::vector<double>> x); // Fit the scaler

        std::vector<std::vector<double>>  fit_transform(std::vector<std::vector<double>> x); //Perform both fit and transform

        std::vector<std::vector<double>> transform(std::vector<std::vector<double>> x); // Perform transformation

};

class MinMaxScaler{

    private:
        std::vector<double> min,max; //Vectors for mean and stddev
        
    public:
        void fit(std::vector<std::vector<double>> x); // Fit the scaler

        std::vector<std::vector<double>>  fit_transform(std::vector<std::vector<double>> x); //Perform both fit and transform

        std::vector<std::vector<double>> transform(std::vector<std::vector<double>> x); // Perform transformation

};

#endif