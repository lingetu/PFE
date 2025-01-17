IFLAGS =   -I../libs_osx_static/opencv/include
CPPFLAGS = -std=c++11 -O3 
CXX = clang++

GL_LIBS = -framework OpenGL -framework GLUT
OPENCV_LIBS = -L../libs_osx_static/opencv/lib -lopencv_calib3d -lopencv_highgui	-lopencv_shape -lopencv_core -lopencv_imgcodecs -lopencv_stitching -lopencv_dnn -lopencv_imgproc -lopencv_superres -lopencv_features2d -lopencv_ml -lopencv_video -lopencv_flann -lopencv_objdetect -lopencv_videoio -lopencv_hal -lopencv_photo -lopencv_videostab
AUX_LIBS = -L../libs_osx_static/opencv/lib -lavcodec -lavdevice -lavfilter -lavformat -lavutil -lIlmImf -lipp_iw -lippicv -littnotify -llibjasper -llibjpeg -llibpng -llibprotobuf -llibtiff -llibwebp -lswresample -lswscale -lzlib
OSX_LIBS = -framework OpenGL -framework Cocoa -framework QTKit -framework QuartzCore -framework AppKit -framework IOKit -framework OpenCL -framework VideoToolbox -framework AudioToolbox -framework VideoDecodeAcceleration -framework CoreGraphics -framework CoreServices -framework Security -framework Accelerate -framework ImageIO -framework CoreMedia -framework CoreVideo -framework AVFoundation -liconv -lm -llzma -lbz2 -lz

ALL_LIBS = $(GL_LIBS) $(OPENCV_LIBS) $(AUX_LIBS) $(OSX_LIBS)

SRC_FILES = main.cpp stream/CameraStream.cpp analyse/EdgeDetection.cpp modelisation/Transformation.cpp  modelisation/OpenGL.cpp  modelisation/GlutMaster.cpp  modelisation/GlutWindow.cpp physics/Ball.cpp physics/AngleModel.cpp  physics/CollisionDetection.cpp physics/Vector2d.cpp   physics/Wall.cpp 

INC_FILES = stream/CameraStream.h analyse/EdgeDetection.h modelisation/Transformation.h modelisation/OpenGL.h modelisation/GlutMaster.h modelisation/GlutWindow.h physics/Ball.h  physics/AngleModel.h physics/CollisionDetection.h physics/Vector2d.h physics/Vector2d.h  physics/Wall.h

OBJ_FILES = main.o CameraStream.o EdgeDetection.o Transformation.o  OpenGL.o  GlutMaster.o  GlutWindow.o Ball.o AngleModel.o  CollisionDetection.o Vector2d.o  Wall.o

ALL_FILES = $(SRC_FILES) $(INC_FILES)

all : pfe_ra

main : $(ALL_FILES)
	$(CXX) -c $(IFLAGS) $(CPPFLAGS) $(SRC_FILES)

pfe_ra : main
	$(CXX) -o pfe_ra $(OBJ_FILES) $(CPPFLAGS) $(ALL_LIBS)
	rm *.o

clean :
	rm *.o
