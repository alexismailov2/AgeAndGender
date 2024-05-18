cmake_minimum_required(VERSION 3.10)
project(ageDetect)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME}
        AgeGender.cpp)

target_include_directories(${PROJECT_NAME}
        PRIVATE
        json/single_include
        ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        ${OpenCV_LIBS})

