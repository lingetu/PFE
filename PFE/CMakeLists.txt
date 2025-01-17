cmake_minimum_required(VERSION 3.8)
project(aMAZEd)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_STANDARD 11)

# Where to find CMake modules and OpenCV
set(OpenCV_DIR "C:\\OpenCV\\x64\\mingw\\lib")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

set(CMAKE_PREFIX_PATH "C:/freeglut")
set(FREEGLUT_INCLUDE_DIRS "C:/freeglut/include")
set(FREEGLUT_LIBRARY_DIRS "C:/freeglut/lib/x64")
set(FREEGLUT_LIBRARIES freeglut)
link_directories(${FREEGLUT_LIBRARY_DIRS})



find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS} ${FREEGLUT_INCLUDE_DIRS})

add_executable(aMAZEd main.cpp stream/CameraStream.cpp stream/CameraStream.h analyse/EdgeDetection.cpp analyse/EdgeDetection.h
        modelisation/Transformation.cpp modelisation/Transformation.h modelisation/OpenGL.cpp modelisation/OpenGL.h
        modelisation/GlutMaster.cpp modelisation/GlutMaster.h modelisation/GlutWindow.cpp modelisation/GlutWindow.h
        physics/Ball.cpp physics/Ball.h physics/AngleModel.cpp physics/AngleModel.h physics/CollisionDetection.cpp
        physics/CollisionDetection.h physics/Vector2d.cpp physics/Vector2d.h physics/CollisionDetection.cpp physics/CollisionDetection.h
        physics/Vector2d.cpp physics/Vector2d.h physics/Wall.cpp physics/Wall.h)

# add OpenCV libs
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_videoio opencv_calib3d)

# add OpenGL libs
set(OpenGL_LIBS libopengl32.a libglu32.a libfreeglut.a)

# linking
if(WIN32)
    target_link_libraries(aMAZEd wsock32 ws2_32 ${OpenCV_LIBS} ${OpenGL_LIBS})
else()
    target_link_libraries(aMAZEd ${OpenCV_LIBS} ${OpenGL_LIBS})
endif()

