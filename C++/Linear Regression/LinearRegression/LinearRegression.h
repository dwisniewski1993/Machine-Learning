//
// Created by Dominik on 19.03.2024.
//

#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <Eigen/Dense>

class LinearRegression {
public:
    LinearRegression()
    {}

    float OrdinaryLeastSquaresCost(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta);
    std::tuple<Eigen::VectorXd, std::vector<float>> GradientDescent(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::VectorXd theta, float alpha, int iters);
    float RSquared(Eigen::MatrixXd y, Eigen::MatrixXd y_hat);

};



#endif //LINEARREGRESSION_H
