LIB = CameraLink.h
LINKING1 = -I"C:\Program Files\SiliconSoftware\Runtime5.2.1\include" -I"c:\\python27\\Lib\\site-packages\\numpy\\core\\include"
LINKING2 = -L"C:\\Program Files\\SiliconSoftware\\Runtime5.2.1\\bin" -lclsersis -lfglib5 
SRC_OBJ = CameraLink.o clSerial.o clExport.o clSetParameter.o clGetParameters.o

cameraLinkModule.dll:$(SRC_OBJ) $(LIB)
	g++ -shared -fPIC -o cameraLinkModule.dll $(SRC_OBJ) $(LINKING2)
	
clSetParameter.o: clSetParameter.cpp $(LIB)
	g++ -fPIC -c clSetParameter.cpp $(LINKING1) 
	
clGetParameters.o: clGetParameters.cpp $(LIB)
	g++ -fPIC -c clGetParameters.cpp $(LINKING1) 
	
clSerial.o: clSerial.cpp $(LIB)
	g++ -fPIC -c clSerial.cpp $(LINKING1) 

CameraLink.o: CameraLink.cpp $(LIB)
	g++ -fPIC -c CameraLink.cpp $(LINKING1) 

clExport.o: clExport.cpp $(LIB)
	g++ -fPIC -c clExport.cpp $(LINKING1) 
	
.PHONY:clean

clean:
	rm *.so *~ *.o

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