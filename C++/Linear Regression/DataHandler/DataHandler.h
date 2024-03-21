//
// Created by Dominik on 14.03.2024.
//

#ifndef ETL_H
#define ETL_H

#include <iostream>
#include <fstream>
#include <Eigen/Dense>

class DataHandler {
    std::string dataset;
    std::string delimeter;
    bool header;

public:
    DataHandler(std::string data, std::string separator, bool head): dataset(data), delimeter(separator), header(head)
    {}

    std::vector<std::vector<std::string>> readCSV();

    Eigen::MatrixXd CSVtoEigen(std::vector<std::vector<std::string>> dataset, int rows, int cols);

    auto Mean(Eigen::MatrixXd data) -> decltype(data.colwise().mean());
    auto Std(Eigen::MatrixXd data) -> decltype((data.array().square().colwise().sum()/(data.rows()-1)).sqrt());
    Eigen::MatrixXd Normalize(Eigen::MatrixXd data, bool normalizeTarget);

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> TrainTestSplit(Eigen::MatrixXd data, float train_size);

    void Vectortofile(std::vector<float> vector, std::string filename);
    void EigentoFile(Eigen::MatrixXd data, std::string filename);
};

#endif //DataHandler_H
