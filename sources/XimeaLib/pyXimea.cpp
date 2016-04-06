#include "ximea.h"


Camera::Camera(int device) :device(device)
{
  capt = new CaptureCAM_XIMEA();
  open(device);
}

Camera::~Camera(){
  release();
}
  
void Camera::toString(){
  
  if(capt){
    cout << "format: " << capt->format << endl;
    cout << "width: " << capt->width << endl;
    cout << "height: " << capt->height << endl;
    cout << "x offset: " << capt->xoffset<< endl;
    cout << "y offset: " << capt->yoffset<< endl;
  }
}

bool Camera::open(int dev)
{
  if (isOpened() == true) {
    release();
  }
  capt->open(dev);
  return isOpened();
}

bool Camera::isOpened()
{
    if(capt->isopened) {
    return true;
  }else{
    return false;
  }
}

void Camera::release()
{
    capt->close();
}

bool Camera::grab()
{
    return capt->grabFrame();
}

bool Camera::set(int propId, double value)
{
  return capt->setProperty(propId, value);
}

double Camera::get(int propId)
{
  double ival = capt->getProperty(propId);
  cout << "ival>>" << ival << endl;
  return ival;
}

pair<bool,void*> Camera::xiread(){

    if(!grab()){
          return std::make_pair<bool, void*>(false, NULL);
        }else{
          return std::make_pair<bool, void*>(true, capt->image.bp);
          // return capt->image.bp;
        }
}

void Camera::addTrigger(int timeout, bool triggered)
{
    capt->addTrigger(timeout, triggered);
}

