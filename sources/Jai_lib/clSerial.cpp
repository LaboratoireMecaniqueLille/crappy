#include "CameraLink.h"
#include <clser.h>

void CaptureCAM_CL::checkSerialCom(int stat){
    switch(stat){
      case CL_ERR_INVALID_REFERENCE: {
        	sprintf(Data,"Invalid reference(%d)\n",stat);
        	throw string(Data);
        	break;
      }
      case CL_ERR_TIMEOUT: {
        	sprintf(Data,"Timeout(%d)\n",stat);
        	throw string(Data);
        	break;
      }
      case CL_ERR_BUFFER_TOO_SMALL: {
        	sprintf(Data,"The buffer is too small(%d)\n",stat);
        	throw string(Data);
        	break;
      }
      case CL_ERR_ERROR_NOT_FOUND: {
        	sprintf(Data,"Error not found(%d)\n",stat);
        	throw string(Data);
        	break;
      }
      case CL_ERR_INVALID_INDEX: {
        	sprintf(Data,"Invalid index(%d)\n",stat);
        	throw string(Data);
        	break;
      }
      case CL_ERR_OUT_OF_MEMORY: {
        	sprintf(Data,"Out of memory(%d)\n",stat);
        	throw string(Data);
        	break;
      }
      case CL_ERR_PORT_IN_USE: {
        	sprintf(Data,"Port already in use(%d)\n",stat);
        	throw string(Data);
        	break;
      }
      case CL_ERR_UNABLE_TO_LOAD_DLL: {
        	sprintf(Data,"Unable to load dll(%d)\n",stat);
        	throw string(Data);
        	break;
      }
    }
}

void CaptureCAM_CL::serialInit(unsigned int serialIndex){
  try
  { 
    unsigned int * numSerialPorts;
    numSerialPorts = (unsigned int *) malloc(32);

    char *portID= NULL;
    unsigned int bufferSize= 512;
    portID = (char *) malloc(bufferSize);
    if((clSerialInit(serialIndex, &serialRefPtr) == CL_ERR_PORT_IN_USE) || (clSerialInit(serialIndex, &serialRefPtr) == CL_ERR_INVALID_INDEX)){
      return;
    }
    serialRefPtr = (void*) malloc(10);
    if (clSerialInit(serialIndex, &serialRefPtr)  == CL_ERR_NO_ERR)
      printf("PORT %d IS OPENED\n", serialIndex);
    char *buf = NULL;
    unsigned int buflen = 10;
    if (clGetSerialPortIdentifier(serialIndex, buf, &buflen) != CL_ERR_BUFFER_TOO_SMALL)
      return;
    buf = (char*) malloc (buflen);
    if (clGetSerialPortIdentifier(serialIndex, buf, &buflen) == CL_ERR_NO_ERR)
      printf("port %i is identified as %s\n", serialIndex, buf);
    free(buf);
  }
  
  catch (exception& e)
  {
    cout << "An exception occurred. Exception: " << e.what() << '\n';
  }
}

void CaptureCAM_CL::serialWrite(char buffer[]){
    clFlushPort(serialRefPtr);
    unsigned int bufferlen= 20;
    unsigned int serialTimeout = 1000;
//     char * str = "(0x10)\r\n";
//     char *newBuffer = (char *) malloc(strlen(buffer)+strlen(str));
//     strcpy(newBuffer, buffer);
//     strcat(newBuffer,str);
    try{
        checkSerialCom(clSerialWrite(serialRefPtr, buffer, &bufferlen, serialTimeout));
    }catch(string const& error){
        cout << "Write ERROR:" << error << endl;
    }
    char *mybuff= NULL;

    unsigned int numBytes=256;
    mybuff = (char *) malloc(numBytes);
    try{
        checkSerialCom(clSerialRead(serialRefPtr, mybuff, &numBytes, serialTimeout));
    }catch(string const& error){
        free(mybuff);
        cout << "Read ERROR:" << error << endl;
// 	serialWrite(buffer);
	return;
    }
//     char temp[8];
//     strncpy(temp, mybuff, 8);
//     cout << "Camera response:" << temp << endl;
    free(mybuff);
}