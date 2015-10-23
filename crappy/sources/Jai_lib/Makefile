LIB = CameraLink.h
LINKING = -L/opt/SiliconSoftware/Runtime5.2.2/lib/ -lclsersis -lfglib5
SRC_OBJ = CameraLink.o clSerial.o clExport.o clSetParameter.o clGetParameters.o

cllib.so:$(SRC_OBJ) $(LIB)
	g++ -shared -fPIC -o cllib.so $(SRC_OBJ) $(LINKING)
	
clSetParameter.o: clSetParameter.cpp $(LIB)
	g++ -fPIC -c clSetParameter.cpp 
	
clGetParameters.o: clGetParameters.cpp $(LIB)
	g++ -fPIC -c clGetParameters.cpp 
	
clSerial.o: clSerial.cpp $(LIB)
	g++ -fPIC -c clSerial.cpp

CameraLink.o: CameraLink.cpp $(LIB)
	g++ -fPIC -c CameraLink.cpp

clExport.o: clExport.cpp $(LIB)
	g++ -fPIC -c clExport.cpp 
	
.PHONY:clean

clean:
	rm *.so *~ *.o
	
# 	g++ -shared -Wl,-soname,cllib -o cllib.so -fPIC  $(SRC_CPP) $(LINKING)
