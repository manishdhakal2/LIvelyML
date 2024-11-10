#include<iostream>
#include<vector>
#include <cstdlib>

class LinearRegression{
    public:
        float lr; // The Learning Rate
        int epoch; // No. of epochs
        
        std::vector<double> y,w,b; // Y- Target variable , W- Weight matrix, B- Bias Matrix
        std::vector<std::vector<double>> x;// The feature space
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------//
        // Constructor 
        LinearRegression(float lr,int epochs){
            this->lr=lr;
            this->epoch=epochs;
            this->init_weights();
        }
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------//
        void init_weights(){
            this->w.clear();
            this->b.clear();

            // Random initialization for random weights
            for (int i=0;i<this->x[0].size();i++){

                this->w.push_back((std::rand()) / RAND_MAX);
                this->b.push_back(0); // all zeros for bias

            }

        }
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------//
        // Calculates the Matrix Multiplication of 2d with 1d vector

        std::vector <double> dotProduct(std::vector<std::vector<double>> x,std::vector<double> y){
            std::vector<double> result(y.size());
            double sum;
            for(int i=0;i<x.size();i++){
                sum=0;
                for(int j=0;j<x[0].size();j++){
                    sum+=x[i][j]*y[j];
                }
                result[i]=sum;
            }
            return result; 
        }
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------//
        void fit(std::vector<std::vector<double>> x,std::vector <double> y){ 
            this->x=x;
            this->init_weights();
            this->y=y;

            std::vector<double> result(this->y.size()); // Variable to store the result of dw

            std::vector <double> y_pred(this->x.size());
            std::vector<double> dW(this->w.size());
            std::vector<double> dB(this->b.size());

            for (int i=0;i<this->epoch;i++){

                y_pred=this->predict(this->x);
                for (int i=0;i<this->y.size();i++){
                    y_pred[i]=this->y[i]-y_pred[i];
                }

                dW=this->dotProduct(this->x,y_pred);
                dB=y_pred;

                for (int i=0;i<dW.size();i++){
                    dW[i]=dW[i]*(-2)/this->x[0].size();

                    this->w[i]-=(this->lr*dW[i]);
                    dB[i]=dB[i]*(-2)/this->x[0].size();
                    this->w[i]-=(this->lr*dB[i]);

                }

            }


        }
  //-----------------------------------------------------------------------------------------------------------------------------------------------------------------//      
         std::vector <double> predict(std::vector<std::vector<double>> x){
            std::vector<double> result(x.size());
            for(int i=0;i<x.size();i++){
                for(int j=0;j<x[0].size();j++){
                    result[i]+=x[i][j]*this->w[j];
                }
                result[i]+=this->b[i];
            }
            return result;
        }
};
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------//