#include "CameraLink.h"

Camera::Camera(int boardNr,/* unsigned int exposure, unsigned int width, unsigned int height, unsigned int xoffset, unsigned int yoffset,*/ double framespersec) :m_boardNr(boardNr),/* m_exposure(exposure), m_width(width), m_height(height), m_xoffset(xoffset), m_yoffset(yoffset),*/ m_framespersec(framespersec)
{
  last_pic_nr = 0;
  timeout = 4;
  fg= NULL;
  camPort= PORT_A;
  nrOfPicturesToGrab  = GRAB_INFINITE;
  nbBuffers= 4;
  samplePerPixel= 1;
  bytePerSample= 1;
  TriggerMode= 1;
  memHandle =NULL;
  serialRefPtr = NULL;
  m_file = NULL;
  m_width = 2560;
  m_height = 2048;
  m_exposure = 8000;
  m_xoffset = 0;
  m_yoffset = 0;
  
}

Camera::~Camera(){
  close();
}
  
void Camera::toString(){
  
  cout << "board number: " << m_boardNr << endl;
  cout << "width: " << m_width << endl;
  cout << "height: " << m_height << endl;
  cout << "Exposure: " << m_exposure << endl;
  cout << "x offset: " << m_xoffset<< endl;
  cout << "y offset: " << m_yoffset<< endl;
  cout << "FPS: " << m_framespersec<< endl;
}

int Camera::init(const char* file) {
  int isSlave =0;
  if ((fg = Fg_InitEx("MediumLineRGB24", m_boardNr, isSlave)) == NULL) {
    fprintf(stderr, "error in Fg_InitEx: %s\n", Fg_getLastErrorDescription(NULL));
    exit(EXIT_FAILURE);
  }
  m_file = file;
  if(Fg_loadConfig(fg,file)!=FG_OK){
    printf("\nFile config loading failed\n");
    exit(EXIT_FAILURE);
  }
  ComNr=m_boardNr*2;
  serialInit(ComNr);
  
  if(Fg_setParameter(fg,FG_TRIGGERMODE,&TriggerMode,camPort)==FG_OK){
    printf("\nTrig config succeed\n");
  }
  size_t totalBufferSize = m_width * m_height * samplePerPixel * bytePerSample * nbBuffers;
  memHandle = Fg_AllocMemEx(fg, totalBufferSize, nbBuffers);
  if (memHandle == NULL) {
    fprintf(stderr, "error in Fg_AllocMemEx: %s\n", Fg_getLastErrorDescription(fg));
    Fg_FreeGrabber(fg);
    exit(EXIT_FAILURE);
  }
  toString();
  try{
//     if(setExposure(m_exposure)!=FG_OK){throw 0;}
//     setWidth(m_width);
//     setHeight(m_height);
//     setYoffset(m_yoffset);
//     setXoffset(m_xoffset);
    int bitAlignment = FG_LEFT_ALIGNED;
    CHECK(FG_BITALIGNMENT, "FG_BITALIGNMENT", &bitAlignment);
//     CHECK(FG_FRAMESPERSEC, "FG_FRAMESPERSEC",  &m_framespersec);
    CHECK(FG_TIMEOUT, "FG_TIMEOUT", &timeout);
//     startAcquire();
  }catch(string const& error){
    cout << "ERROR: " << error <<endl;
    close();
//     init(file);
  }catch(int e){
    close();
//     init(file);
  }
  return FG_OK;
}

int Camera::startAcquire(){
  if ((Fg_AcquireEx(fg, camPort, nrOfPicturesToGrab, ACQ_STANDARD, memHandle)) < 0) {
    fprintf(stderr, "Fg_AcquireEx() failed: %s\n", Fg_getLastErrorDescription(fg));
    Fg_FreeMemEx(fg, memHandle);
    Fg_FreeGrabber(fg);
    return FG_ERROR;
  }
  return FG_OK;
}

int Camera::restart(){
  if ((Fg_AcquireEx(fg, camPort, nrOfPicturesToGrab, ACQ_STANDARD, memHandle)) < 0) {
    fprintf(stderr, "Fg_AcquireEx() failed: %s\n", Fg_getLastErrorDescription(fg));
    Fg_FreeMemEx(fg, memHandle);
    Fg_FreeGrabber(fg);
    return FG_ERROR;
  }
  serialInit(ComNr);
  return FG_OK;
}

void *Camera::getBuffer(){
  try{
    cur_pic_nr = Fg_getLastPicNumberBlockingEx(fg, last_pic_nr + 1, camPort, timeout, memHandle);
    if (cur_pic_nr < 0) {
      sprintf(Data,"Fg_getLastPicNumberBlockingEx(%li) failed: %s\n",last_pic_nr+1, Fg_getLastErrorDescription(fg));
//       cout << "timeout: " << timeout << endl;
      stop();
      throw string(Data);
    }else{
      last_pic_nr = cur_pic_nr;
//       timeout=4;
      void *ImgPtr = Fg_getImagePtrEx(fg, last_pic_nr, camPort, memHandle);
      return ImgPtr;
    }
  }catch(string const& error){
    cout <<"ERROR: " << error << endl;
    sleep(5);
    timeout+= 100;
    last_pic_nr = 0;
    init(m_file);
    startAcquire();
    return getBuffer();
  }
}
void Camera::stop(){
  Fg_stopAcquire(fg,camPort);
}

void Camera::close(){
  Fg_stopAcquire(fg,camPort);
  Fg_FreeMemEx(fg, memHandle);
  Fg_FreeGrabber(fg);
  free(serialRefPtr);
}
