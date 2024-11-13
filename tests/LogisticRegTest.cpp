#include<iostream>
#include<vector>
#include "../include/preprocessing.hpp"

int main(){
    LogisticRegression model=LogisticRegression

    std::vector<std::vector<double>> data={{100,50},{200,60},{300,70},{400,80}};
    model.fit(data);
    std::vector<std::vector<double>> result=model.transform(data);
    for(int i=0;i<4;i++){
        for(int j=0;j<2;j++){
        std::cout<<result[i][j]<<" ";
    }
    std::cout<<std::endl;
    }
}