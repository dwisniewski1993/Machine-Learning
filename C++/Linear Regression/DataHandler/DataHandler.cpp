//
// Created by Dominik on 14.03.2024.
//
#include "DataHandler.h"
#include <vector>
#include <stdlib.h>
#include <cmath>
#include <string>
#include <boost/algorithm/string.hpp>
#include <sstream>

std::vector<std::vector<std::string>> DataHandler::readCSV() {
    std::ifstream file(dataset);
    std::vector<std::vector<std::string>> dataString;

    std::string line;

    while(std::getline(file, line)) {
        std::vector<std::string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
        dataString.push_back(vec);
    }

    file.close();

    return dataString;
}

bool isNumber(const std::string& str) {
    char* end;
    std::strtod(str.c_str(), &end);
    return end != str.c_str() && *end == '\0';
}

std::string concatenateLetters(const std::string& str) {
    std::stringstream result;
    for (char c : str) {
        result << static_cast<int>(c);
    }
    return result.str();
}

Eigen::MatrixXd DataHandler::CSVtoEigen(std::vector<std::vector<std::string>> dataset, int rows, int cols) {

    if (header==true) {
        rows = rows - 1;
    }

    Eigen::MatrixXd mat(cols, rows);
    for (int i = 0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            if (isNumber(dataset[i][j])) {
                mat(j, i) = atof(dataset[i][j].c_str());
            }
            else {
                mat(j, i) = atof(concatenateLetters(dataset[i][j]).c_str());
            }
        }
    }

    return mat.transpose();
}

std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> DataHandler::TrainTestSplit(Eigen::MatrixXd data, float train_size){

    int rows = data.rows();
    int train_rows = round(train_size*rows);
    int test_rows = rows - train_rows;

    Eigen::MatrixXd train = data.topRows(train_rows);

    Eigen::MatrixXd X_train = train.leftCols(data.cols()-1);
    Eigen::MatrixXd y_train = train.rightCols(1);

    Eigen::MatrixXd test = data.bottomRows(test_rows);

    Eigen::MatrixXd X_test = test.leftCols(data.cols()-1);
    Eigen::MatrixXd y_test = test.rightCols(1);

    return std::make_tuple(X_train, y_train, X_test, y_test);
}

auto DataHandler::Mean(Eigen::MatrixXd data) -> decltype(data.colwise().mean()){
    return data.colwise().mean();
}

auto DataHandler::Std(Eigen::MatrixXd data) -> decltype(((data.array().square().colwise().sum())/(data.rows()-1)).sqrt()){
    return ((data.array().square().colwise().sum())/(data.rows()-1)).sqrt();
}

Eigen::MatrixXd DataHandler::Normalize(Eigen::MatrixXd data, bool normalizeTarget){

    Eigen::MatrixXd dataNorm;
    if(normalizeTarget==true) {
        dataNorm = data;
    } else {
        dataNorm = data.leftCols(data.cols()-1);
    }

    auto mean = Mean(dataNorm);
    Eigen::MatrixXd scaled_data = dataNorm.rowwise() - mean;
    auto std = Std(scaled_data);

    Eigen::MatrixXd norm = scaled_data.array().rowwise()/std;

    if(normalizeTarget==false) {
        norm.conservativeResize(norm.rows(), norm.cols()+1);
        norm.col(norm.cols()-1) = data.rightCols(1);
    }

    return norm;
}

void DataHandler::Vectortofile(std::vector<float> vector, std::string filename){
    std::ofstream output_file(filename);
    std::ostream_iterator<float> output_iterator(output_file, "\n");
    std::copy(vector.begin(), vector.end(), output_iterator);
}

void DataHandler::EigentoFile(Eigen::MatrixXd data, std::string filename){
    std::ofstream output_file(filename);
    if(output_file.is_open()){
        output_file << data << "\n";
    }
}