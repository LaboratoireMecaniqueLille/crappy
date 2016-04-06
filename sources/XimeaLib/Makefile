LIB = ximea.h
LINKING1 = -Ic:\\XIMEA\\API 
LINKING2 = -Lc:\\XIMEA\\API\\x64 -lm3apiX64
SRC_OBJ = ximea.o pyximea.o ximeaExport.o

ximeaModule.dll:$(SRC_OBJ) $(LIB)
	g++ -shared -fPIC -std=c++11 -o ximeaModule.dll $(SRC_OBJ) $(LINKING2)
	
pyximea.o: pyximea.cpp $(LIB)
	g++ -c pyximea.cpp $(LINKING1)

ximea.o: ximea.cpp $(LIB)
	g++ -c ximea.cpp $(LINKING1)

ximeaExport.o: ximeaExport.cpp $(LIB)
	g++ -c ximeaExport.cpp $(LINKING1)
	
.PHONY:clean

clean:
	del *.dll *~ *.o

# 	g++ -shared -Wl,-soname,cllib -o cllib.so -fPIC  $(SRC_CPP) $(LINKING)
#
#LIB = CameraLink.h
#LINKING = -L/opt/SiliconSoftware/Runtime5.4.1.1/lib64/ -lclsersis -lfglib5
#SRC_OBJ = CameraLink.o clSerial.o clExport.o clSetParameter.o clGetParameters.o
#
#cameraLinkModule.so:$(SRC_OBJ) $(LIB)
#	g++ -shared -fPIC -o cameraLinkModule.so $(SRC_OBJ) $(LINKING)
#	
#clSetParameter.o: clSetParameter.cpp $(LIB)
#	g++ -fPIC -c clSetParameter.cpp 
#	
#clGetParameters.o: clGetParameters.cpp $(LIB)
#	g++ -fPIC -c clGetParameters.cpp 
#	
#clSerial.o: clSerial.cpp $(LIB)
#	g++ -fPIC -c clSerial.cpp
#
#CameraLink.o: CameraLink.cpp $(LIB)
#	g++ -fPIC -c CameraLink.cpp
#
#clExport.o: clExport.cpp $(LIB)
#	g++ -fPIC -c clExport.cpp 
#	
#.PHONY:clean
#
#clean:
#	rm *.so *~ *.o