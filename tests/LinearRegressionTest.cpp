#include<iostream>
#include "../include/LinearModels.hpp"
#include<vector>

int main(){
    std::cout << "Entering main function..." << std::endl;

    LinearRegression model=LinearRegression(0.001,1000);
    std::cout<<"Test 1";

    std::vector<std::vector<double>>x={{10},{20},{30},{40},{50},{60},{70},{80}};
    std::vector<double> y={110,210,310,410,510,610,710,810};

    model.fit(x,y);
    x.clear();
    x={{90},{100},{110},{120},{130}};
    y.clear();
    y={910,1010,1110,1210,1310};

    std::vector<double> y_pred(y.size());
    y_pred=model.predict(x);

    for(int i=0;i<5;i++){
        std::cout<<y_pred[i]<<std::endl;
    }
    


}
