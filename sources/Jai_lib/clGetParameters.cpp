#include "CameraLink.h"

unsigned int Camera::getExposure(){
    return m_exposure;
}

unsigned int Camera::getWidth(){
  return m_width;
}

unsigned int Camera::getHeight(){
  return m_height;
}

unsigned int Camera::getXoffset(){
  return m_xoffset;
}

unsigned int Camera::getYoffset(){
  return m_yoffset;
}
