cmake_minimum_required(VERSION 3.28)
project(Linear_Regression)

set(CMAKE_CXX_STANDARD 17)

add_executable(Linear_Regression main.cpp
        DataHandler/DataHandler.cpp
        DataHandler/DataHandler.h
        LinearRegression/LinearRegression.cpp
        LinearRegression/LinearRegression.h)
