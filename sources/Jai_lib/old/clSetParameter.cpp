#include "CameraLink.h"


int Camera::setExposure(unsigned int t_exposure){
  try{
      CHECK(FG_EXPOSURE, "FG_EXPOSURE", &t_exposure);
      m_exposure = t_exposure;
      clFlushPort(serialRefPtr);
      sprintf(Data,"PE=%d(0x10)\r\n",m_exposure); 
      cout << "Setting Exposure parameter: " << Data;
      serialWrite(Data);
      return FG_OK;
  }catch(string const& error){
      cout << "ERROR: " <<  error << endl;
      return FG_ERROR;
  }
}

int Camera::setWidth(unsigned int t_width){
  try{
      CHECK(FG_WIDTH, "FG_WIDTH", &t_width);
      m_width = t_width;
      clFlushPort(serialRefPtr);
      sprintf(Data,"WTC=%d(0x10)\r\n",m_width); 
      cout << "setting Width parameter: " << Data;
      serialWrite(Data);
      return FG_OK;
  }catch(string const& error){
      cout << "ERROR: " <<  error << endl;
      return FG_ERROR;
  }
}

int Camera::setHeight(unsigned int t_height){
  try{
    CHECK(FG_HEIGHT, "FG_HEIGHT", &t_height);
    m_height = t_height;
    clFlushPort(serialRefPtr);
    sprintf(Data,"HTL=%d(0x10)\r\n",m_height);
    cout << "Setting height parameter:" << Data;
    serialWrite(Data);
    return FG_OK;
  }catch(string const& error){
    cout << "ERROR: " <<  error << endl;
    return FG_ERROR;
}
}

int Camera::setXoffset(unsigned int t_xoffset){
  try{
    m_xoffset = t_xoffset;
      CHECK(FG_XOFFSET, "FG_XOFFSET", &t_xoffset);
    clFlushPort(serialRefPtr);
    sprintf(Data,"OFC=%d(0x10)\r\n",m_xoffset); 
    cout << "Setting X offset parameter: " << Data;
    serialWrite(Data);
    return FG_OK;
  }catch(string const& error){
    cout << "ERROR: " <<  error << endl;
    return FG_ERROR;
}
}
int Camera::setYoffset(unsigned int t_yoffset){
  try{
    m_yoffset = t_yoffset;
    CHECK(FG_YOFFSET, "FG_YOFFSET", &t_yoffset);
    clFlushPort(serialRefPtr);
    sprintf(Data,"OFL=%d(0x10)\r\n",m_yoffset); 
    cout << "Setting Y offset parameter:  " << Data;
    serialWrite(Data);
    return FG_OK;
  }catch(string const& error){
    cout << "ERROR: " <<  error << endl;
    return FG_ERROR;
}
}
