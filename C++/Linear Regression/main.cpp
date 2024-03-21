#include <iostream>
#include "DataHandler/DataHandler.h"
#include <string>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <filesystem>
#include <unistd.h>
#include <libgen.h>
#include "LinearRegression/LinearRegression.h"


int main(int argc, char* argv[] )
{
    std::cout << "Linear Regression" << std::endl;
    bool header = false;
    std::string file;

    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        char* dir = dirname(cwd);

        file = std::string(dir) + "\\datasets\\train.csv";

    } else {
        return 1;
    }

    DataHandler data_handler(file, ",", header);
    LinearRegression linear_regression;

    std::vector<std::vector<std::string>> dataset = data_handler.readCSV();

    int rows = dataset.size();
    int cols = dataset[0].size();

    Eigen::MatrixXd dataMat = data_handler.CSVtoEigen(dataset, rows, cols);
    Eigen::MatrixXd norm = data_handler.Normalize(dataMat, true);

    Eigen::MatrixXd X_train, y_train, X_test, y_test;
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> split_data = data_handler.TrainTestSplit(dataMat, 0.8);
    std::tie(X_train, y_train, X_test, y_test) = split_data;

    Eigen::VectorXd vec_train = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd vec_test = Eigen::VectorXd::Ones(X_test.rows());

    X_train.conservativeResize(X_train.rows(), X_train.cols()+1);
    X_train.col(X_train.cols()-1) = vec_train;

    X_test.conservativeResize(X_test.rows(), X_test.cols()+1);
    X_test.col(X_test.cols()-1) = vec_test;

    Eigen::VectorXd theta = Eigen::VectorXd::Ones(X_train.cols());
    float alpha = 0.01;
    int iters = 5;

    Eigen::VectorXd thetaOut;
    std::vector<float> cost;

    std::tuple<Eigen::VectorXd, std::vector<float>> gradient_descent = linear_regression.GradientDescent(X_train, y_train, theta, alpha, iters);
    std::tie(thetaOut, cost) = gradient_descent;

    std::cout << "DataMat: " << dataMat << std::endl;

    auto mu_data = data_handler.Mean(dataMat);
    auto mu_z = mu_data(0,dataMat.cols()-1);

    auto scaled_data = dataMat.rowwise() - dataMat.colwise().mean();

    std::cout << "Scaled data: " << scaled_data<< std::endl;

    auto sigma_data = data_handler.Std((scaled_data));
    auto sigma_z = sigma_data(0,scaled_data.cols()-1);

    Eigen::MatrixXd y_train_hat = (X_train*thetaOut*sigma_z).array() + mu_z;
    Eigen::MatrixXd y = dataMat.col(5).topRows(1339);

    float R_Squared = linear_regression.RSquared(y,y_train_hat);
    std::cout << "R-Squared: " << R_Squared << std::endl;

    return EXIT_SUCCESS;
}
