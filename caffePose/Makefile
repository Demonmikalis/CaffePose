#----------------------------------------------------------
#         Copyright: MR.Black <demonmikalis@google.com>
#         2018-2019 Engineering Department
#         Main Make File For Machine Learning App         
#                   
#                 
#   Use of this software is governed by a License Agreement   
#    ** See the file License for the Conditions of Use  **    
#    **     This banner notice must not be removed      **    
#                                                             
# ----------------------------------------------------------- 
# File: Makefile.  Generated from Makefile.in by configure.
# ----------------------------------------------------------- 

CPP = g++
CPPFLAGS = -Wall -g -std=c++11
LDFLAGS = -L/usr/local/lib -L/usr/local/cuda-8.0/lib64 \
-lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d \
-lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc \
-lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect \
 -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres \
-lopencv_ts -lopencv_video -lopencv_videostab -lcufft \
-lnpps -lnppi -lnppc -lcudart -lrt -lpthread -lm -ldl -lcaffe -lglog -lgflags\
-lboost_system -lopenpose

INC = -I/usr/local/include -I/usr/local/cuda-8.0/include

LDFLAGS2 = -L/usr/local/lib\
-lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d \
-lopencv_flann  -lopencv_highgui -lopencv_imgproc \
-lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect \
 -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres \
-lopencv_ts -lopencv_video -lopencv_videostab -lm -ldl -lcaffe -lglog -lgflags\
-lboost_system 

INC2 = -I/usr/local/include -I/usr/local/cuda-8.0/include

object = classification.o array.o point.o errorAndLog.o netCaffe.o nmsCaffe.o resizeAndMergeCaffe.o \
bodyPartConnectorCaffe.o rectangle.o
 
all: main openpose single

openpose: openpose.o
	$(CPP) -o $@ $(CPPFLAGS) $^ $(LDFLAGS) $(INC)

single: extractFromImage.o
	$(CPP) -o $@ $(CPPFLAGS) $^ $(LDFLAGS) $(INC)

main: $(object) 
	$(CPP) -o $@ $(CPPFLAGS) $^ $(LDFLAGS2) $(INC2)

%.o: %.cpp
	$(CPP) $(CPPFLAGS) -c -o $@ $< $(INC)

clean:
	rm *.o openpose single main

.PHONY: all



