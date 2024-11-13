#include<iostream>
#include<vector>
#include<cmath>
#include "../include/preprocessing.hpp"


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------//

void StandardScaler::fit( std::vector<std::vector<double>> x){
;
    
    this->mean.assign(x[0].size(),0.0); // Initialize all the means and standard deviations to zero
    this->stdev.assign(x[0].size(),0.0);
    

    for(int i=0;i<x.size();i++){
        for(int j=0;j<x[0].size();j++){
            this->mean[j]+=x[i][j]; //Calculate the sum of individual features
        }
    }
    for(int i=0;i<this->mean.size();i++){
        this->mean[i]/=x.size(); // Divide each sum by no of features
    }
    for(int i=0;i<x.size();i++){
        for(int j=0;j<x[0].size();j++){
            this->stdev[j]+=((x[i][j]-this->mean[j])*(x[i][j]-this->mean[j])); // Calculate (x-mean)^2
        }
    }
    for(int i=0;i<this->mean.size();i++){
        this->stdev[i]=std::sqrt(this->stdev[i]/x.size());// Calculate sqrt(x-mean^2/N)
        if (this->stdev[i] == 0) {
                this->stdev[i] = 1.0;  // Prevent division by zero (if the feature is constant)
            } 
    }

}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------//

std::vector<std::vector<double>> StandardScaler::transform(std::vector<std::vector<double>> x){
    std::vector<std::vector<double>> result(x.size(),std::vector<double>(x[0].size()));
    const double epsilon=1e-8;
    
    for(int i=0;i<x.size();i++){
        for(int j=0;j<x[0].size();j++){
            result[i][j]=(x[i][j]-this->mean[j])/(this->stdev[j]+epsilon); // Perform (x-mean)/sttdev for each feature

        }
    }
    return result;
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------//

std::vector<std::vector<double>> StandardScaler::fit_transform(std::vector<std::vector<double>> x){

    this->fit(x);
    return this->transform(x); //return transformed data

}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------//


void MinMaxScaler::fit( std::vector<std::vector<double>> x){

    
    this->min.assign(x[0].size(),0.0); // Initialize all the means and standard deviations to zero
    this->max.assign(x[0].size(),0.0);

    

    for(int i=0;i<x[0].size();i++){
        this->min[i]=x[0][i]; //Assign initial value
    }

    for(int i=0;i<x.size();i++){
        for(int j=0;j<x[0].size();j++){
            if(x[i][j]>max[j]){
                this->max[j]=x[i][j]; // get the maximum
            } 
            if(x[i][j]<this->min[j]){
                this->min[j]=x[i][j]; //get the minimum
            }
        }
    }
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------//

std::vector<std::vector<double>> MinMaxScaler::transform(std::vector<std::vector<double>> x){
    std::vector<std::vector<double>> result(x.size(),std::vector<double>(x[0].size()));
    const double epsilon=1e-8; //Small Value for negating zero division error
    
    for(int i=0;i<x.size();i++){
        for(int j=0;j<x[0].size();j++){
            result[i][j]=(x[i][j]-this->min[j])/(this->max[j]-this->min[j]+epsilon); // Perform (x-min)/(max-min)
        }
    }
    return result;
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------//

std::vector<std::vector<double>> MinMaxScaler::fit_transform(std::vector<std::vector<double>> x){

    this->fit(x);
    return this->transform(x); //return transformed data

}


