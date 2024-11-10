#include<iostream>
#include<vector>
#include <cstdlib>
#include "../include/LinearModels.hpp"

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------//
        // Constructor 
        LinearRegression::LinearRegression(double lr,int epochs){
            this->lr=lr;
            this->epoch=epochs;
        }
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------//
        void LinearRegression::init_weights(){
            this->w.clear();
            this->b=0;

            // Random initialization for random weights
            for (int i=0;i<this->x[0].size();i++){

                this->w.push_back((std::rand()) / RAND_MAX); 


            }



        }

        double LinearRegression::MSE(std::vector<double> y,std::vector <double> y_pred){
            double error=0;
            for(int i=0;i<y.size();i++){
                error+=(y_pred[i]-y[i])*(y_pred[i]-y[i]);
            }
            return error;
        }
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------//
        // Calculates the Matrix Multiplication of 2d with 1d vector

        std::vector <double> LinearRegression::dotProduct(std::vector<std::vector<double>> x,std::vector<double> y){
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
        void LinearRegression::fit(std::vector<std::vector<double>> x,std::vector <double> y){ 
            this->x=x;
            this->init_weights();
            this->y=y;

            std::vector<double> result(this->y.size()); // Variable to store the result of dw

            std::vector <double> y_pred(this->x.size());
            std::vector<double> dW(this->w.size());
            double dB=0;

            for (int i=0;i<this->epoch;i++){

                if(i%100==0){
                    std::cout<<"Epoch "<<i<<std::endl;
                    std::cout<<"Training Loss "<<this->MSE(y_pred,this->y)<<std::endl;
                }
                dB=0;

                y_pred=this->predict(this->x);
                for (int i=0;i<this->y.size();i++){
                    y_pred[i]=this->y[i]-y_pred[i];
                }

                dW=this->dotProduct(this->x,y_pred);
                for(int i=0;i<y_pred.size();i++){
                    dB+=y_pred[i];
                }
                dB=dB*(-2)/this->x[0].size();
                this->b-=(this->lr*dB);

                for (int i=0;i<dW.size();i++){
                    dW[i]=dW[i]*(-2)/this->x[0].size();

                    this->w[i]-=(this->lr*dW[i]);
                    
                    

                }

            }


        }
  //-----------------------------------------------------------------------------------------------------------------------------------------------------------------//      
         std::vector <double> LinearRegression::predict(std::vector<std::vector<double>> x){

            std::vector<double> result(x.size());
            for(int i=0;i<x.size();i++){
                for(int j=0;j<x[0].size();j++){
                    result[i]+=x[i][j]*this->w[j];
                }
                result[i]+=this->b;
            }
            return result;
        }
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------//