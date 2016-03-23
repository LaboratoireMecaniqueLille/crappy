/** @addtogroup sources
 *  @{
 */

/** @addtogroup cameralink 
 *  @{
 */

/** @addtogroup capturecam 
 *  @{
 */


/** 
 * \file CameraLink.cpp
 * \brief CaptureCAM_CL class
 * 
 * This class allows to parameter and control a camera device throught a cameraLink interface
 * \author Robin Siemiatkowski
 * \version 0.1
 * \date 29/02/2016
 */

#include "CameraLink.h"
/**
 * \fn CaptureCAM_CL::CaptureCAM_CL()
 * \brief Constructor of CaptureCAM_CL Class
 *        It Initialize the camera class
 * 
 */
CaptureCAM_CL::CaptureCAM_CL() {
	init();
}


/**
 * \fn CaptureCAM_CL::~CaptureCAM_CL()
 * \brief Destructor of CaptureCAM_CL Class
 * 
 */
CaptureCAM_CL::~CaptureCAM_CL(){
	close();
}

/**
 * \fn void CaptureCAM_CL::init()
 * \brief This function initialize the CaptureCAM_CL attributes.
 * 
 */
void CaptureCAM_CL::init()
{
    isopened=false;
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
    file = NULL;
    width = 640;
    height = 513;
    exposure = 8000;
    xoffset = 0;
    yoffset = 0;
    framespersec = 99;
    nId = 0;
}

/*void CaptureCAM_CL::display(int dis){
    if(dis==1){
      if(nId== 0)
	  nId=CreateDisplay(24,640,513);
      cout << "nId: " << nId << endl;
      SetBufferWidth(nId, 640, 513);
      DrawBuffer(nId, ImgPtr, 1, 0);
    }else{
      if(nId!=0){
	CloseDisplay(nId);
	nId=0;
      }
    }
}*/

/**
 * \fn bool CaptureCAM_CL::open( int wIndex, const char* file)
 * \brief Initialize camera input.
 *
 * \param wIndex Number of the camera device, starts at 0.
 * \param file path to the configuration file of the camera, cannot be Null.
 * \return True if the camera was correctly openned.
 */
bool CaptureCAM_CL::open( int wIndex, const char* file)
{
    int isSlave =0;
    boardNr=wIndex;
    if ((fg = Fg_InitEx("FullLineGray8", wIndex, isSlave)) == NULL) {
      fprintf(stderr, "error in Fg_InitEx: %s\n", Fg_getLastErrorDescription(NULL));
      exit(EXIT_FAILURE);
    }
    if(Fg_loadConfig(fg,file)!=FG_OK){
      printf("\nFile config loading failed\n");
      exit(EXIT_FAILURE);
    }
    ComNr=boardNr*2;
    serialInit(ComNr);
  
    if(Fg_setParameter(fg,FG_TRIGGERMODE,&TriggerMode,camPort)==FG_OK){
      printf("\nTrig config succeed\n");
    }
    size_t totalBufferSize = width * height * samplePerPixel * bytePerSample * nbBuffers;
    memHandle = Fg_AllocMemEx(fg, totalBufferSize, nbBuffers);
    if (memHandle == NULL) {
        fprintf(stderr, "error in Fg_AllocMemEx: %s\n", Fg_getLastErrorDescription(fg));
        Fg_FreeGrabber(fg);
        exit(EXIT_FAILURE);
    }
    int mvret = FG_OK;
    mvret = Fg_setParameter(fg, FG_WIDTH, &width, camPort);
    HandleResult(mvret, "error while setting width");
    mvret = Fg_setParameter(fg, FG_HEIGHT, &height, camPort);
    HandleResult(mvret, "error while setting height");
    format = FG_GRAY16;
    mvret = Fg_setParameter(fg,  FG_FORMAT, &format , camPort);
    HandleResult(mvret, "error while setting data format");
    
    timeout = 10000;
    
    return true;
}



/**
 * \fn void CaptureCAM_CL::stop()
 * \brief Stop the acquisition of a camera device.
 */
int CaptureCAM_CL::stop(){
  if(Fg_stopAcquire(fg,camPort)!=FG_OK){
      cout << "Stop acquisition failed: " << Fg_getLastErrorDescription(fg)<< endl;
      return FG_ERROR;
  }
  isopened=false;
  return FG_OK;
}

/**
 * \fn void CaptureCAM_CL::close()
 * \brief Close the camera device: stop the acquisition and free the allocated memory.
 */
void CaptureCAM_CL::close(){
  Fg_stopAcquire(fg,camPort);
  Fg_FreeMemEx(fg, memHandle);
  Fg_FreeGrabber(fg);
  isopened=false;
  free(serialRefPtr);
}


/**
 * \fn bool CaptureCAM_CL::grabFrame()
 * \brief Grab a frame, the data pointer is stored in the ImgPtr attribute.
 * \return True if the frame grabber has successfully grabbed a frame.
 */
bool CaptureCAM_CL::grabFrame()
{
    try{
    cur_pic_nr = Fg_getLastPicNumberBlockingEx(fg, last_pic_nr + 1, camPort, timeout, memHandle);
    if (cur_pic_nr < 0) {
      sprintf(Data,"Fg_getLastPicNumberBlockingEx failed: %s\n", Fg_getLastErrorDescription(fg));
//       cout << "timeout: " << timeout << endl;
      stop();
      throw string(Data);
    }else{
      last_pic_nr = cur_pic_nr;
//       timeout=4;
      ImgPtr = Fg_getImagePtrEx(fg, last_pic_nr, camPort, memHandle);
      return true;
    }
  }catch(string const& error){
    cout <<"ERROR: " << error << endl;
    sleep(5);
    timeout+= 100;
    last_pic_nr = 0;
    init();
    open(boardNr,file);
    startAcquire();
    return grabFrame();
  }
}


/**
 * \fn void CaptureCAM_CL::resetCvImage()
 * \brief reset the width, height and format attributes.
 *
 */
void CaptureCAM_CL::resetCvImage()
{
  //TODO
//     if( (int)image.width != width || (int)image.height != height || image.frm != (XI_IMG_FORMAT)format)
//     {
// 	xiGetParamInt( hmv, XI_PRM_WIDTH, &width);
// 	xiGetParamInt( hmv, XI_PRM_HEIGHT, &height);
// 	xiGetParamInt( hmv, XI_PRM_IMAGE_DATA_FORMAT, &format);
//     }
}

/**
 * \fn int CaptureCAM_CL::getProperty( int property_id )
 * \brief Reading the current value of a parameter from a frame grabber. 
 *
 * \param property_id As argument, a identification number is needed.
    If the identification number is unknown, the parameter name has to be given.
    It can be one of the following:
        - FG_CAMSTAUS: If a camera signal is on CameraLink port value is 1 else 0.
        - FG_REVNR: Current revision version of camera DLL.
        - FG_TIMEOUT: Time in seconds until device driver displays a timeout of the frame grabber. 
        - FG_WIDTH: Width of the clipping image.
        - FG_MAXWIDTH: Maximum width of the clipping image.
        - FG_HEIGHT: Height of the clipping image.
        - FG_MAXHEIGHT: Maximum height of the clipping image.
        - FG_XSHIFT: Number of invalid words at the beginning of a row (modulo of the width of the interface).
        - FG_XOFFSET: X-offset from the left top corner in pixel.
        - FG_YOFFSET: Y-offset from the left top corner in pixel. 
        - FG_FRAMESPERSEC: Number of images per second.
        - FG_EXPOSURE: Exposure time in µs.
        - FG_FORMAT: Color format of the transferred image
                        -# 8bit gray (FG_GRAY)
                        -# 16bit color (FG_GRAY16)
                        -# 24bit color (FG_COL24).
                     See color management of the according frame grabber design. 
        - FG_PORT: Logical number of the active CameraLink port.
        - FG_PIXELDEPTH: Returns the depth of color of the pixel.
        - FG_LINEALIGNMENT: Returns the alignment of a line (in bits).
        - FG_TRANSFER_LEN: Returns the length of the last DMA transfer.
        - FG_TRIGGERMODE: Trigger modes:
                            -# FREE_RUN
                            -# GRABBER_CONTROLLED
                            -# GRABBER_CONTROLLED_SYNCRON
                            -# ASYNC_SOFTWARE_TRIGGER
                            -# ASYNC_TRIGGER. 
        - FG_STROBPULSEDELAY: Strobe delay to the trigger in µs.
        - FG_TWOCAMMODEL: Returns the value, if the loaded camera applet is a dual (1) or a single applet (0).
        - FG_HDSYNC: Returns the HDSYNC value. 
        - FG_GLOBAL_ACCESS: Returns the value for the set plausibility access.
        - FG_BOARD_INFORMATION: Information on the board type:
                                    -# BINFO_BOARD_TYPE:
                                        - 0xa40 for microEnable IV-Base x1
                                        - 0xa41 for microEnable IV-Full x1
                                        - 0xa44 for microEnable IV-Full x4

                                    -# BINFO_POCL:
                                        - 0 for microEnable IV-Base x1
                                        - 1 for microEnable IV-Base x1 PoCL
 * \return 
        - The value of the parameter.
 */
unsigned int CaptureCAM_CL::getProperty( int property_id )
{
    if(fg == NULL)
        return 0;
    unsigned int value = NULL;
    switch( property_id )
    {
    case FG_PORT                : Fg_getParameter( fg, FG_PORT, &value, camPort); break;
    case FG_WIDTH               : Fg_getParameter( fg, FG_WIDTH, &value, camPort); break;
    case FG_HEIGHT              : Fg_getParameter( fg, FG_HEIGHT, &value, camPort); break;
    case FG_XOFFSET             : Fg_getParameter( fg, FG_XOFFSET, &value, camPort); break;
    case FG_YOFFSET             : Fg_getParameter( fg, FG_YOFFSET, &value, camPort); break;
    case FG_XSHIFT              : Fg_getParameter( fg, FG_XSHIFT, &value, camPort); break;
    case FG_TIMEOUT             : Fg_getParameter( fg, FG_TIMEOUT, &value, camPort); break;
    case FG_FRAMESPERSEC        : Fg_getParameter( fg, FG_FRAMESPERSEC, &value, camPort); break;
    case FG_FORMAT              : Fg_getParameter( fg, FG_FORMAT, &value, camPort); break;
    case FG_EXPOSURE            : Fg_getParameter( fg, FG_EXPOSURE, &value, camPort); break;
    case FG_TRIGGERMODE         : Fg_getParameter( fg, FG_TRIGGERMODE, &value, camPort); break;
    case FG_STROBEPULSEDELAY    : Fg_getParameter( fg, FG_STROBEPULSEDELAY, &value, camPort); break;
    case FG_GLOBAL_ACCESS       : Fg_getParameter( fg, FG_GLOBAL_ACCESS, &value, camPort); break;
    case FG_CAMSTATUS           : Fg_getParameter( fg, FG_CAMSTATUS, &value, camPort); break;
    case FG_REVNR               : Fg_getParameter( fg, FG_REVNR, &value, camPort); break;
    case FG_MAXHEIGHT           : Fg_getParameter( fg, FG_MAXHEIGHT, &value, camPort); break;
    case FG_PIXELDEPTH          : Fg_getParameter( fg, FG_PIXELDEPTH, &value, camPort); break;
    case FG_LINEALIGNMENT       : Fg_getParameter( fg, FG_LINEALIGNMENT, &value, camPort); break;
    case FG_TRANSFER_LEN        : Fg_getParameter( fg, FG_TRANSFER_LEN, &value, camPort); break;
    case FG_TWOCAMMODEL         : Fg_getParameter( fg, FG_TWOCAMMODEL, &value, camPort); break;
    case FG_HDSYNC              : Fg_getParameter( fg, FG_HDSYNC, &value, camPort); break;
    case FG_BOARD_INFORMATION   : Fg_getParameter( fg, FG_BOARD_INFORMATION, &value, camPort); break;

    }
    return value;
}


/**
 * \fn bool CaptureCAM_CL::setProperty( int property_id, int value )
 * \brief Setting the value of a parameter from a frame grabber. 
 *
 * \param property_id As argument, a identification number is needed.
    If the identification number is unknown, the parameter name has to be given.
    It can be one of the following:
        - FG_TIMEOUT: Time in seconds until device driver displays a timeout of the frame grabber. 
        - FG_WIDTH: Width of the clipping image.
        - FG_HEIGHT: Height of the clipping image.
        - FG_XSHIFT: Number of invalid words at the beginning of a row (modulo of the width of the interface).
        - FG_XOFFSET: X-offset from the left top corner in pixel.
        - FG_YOFFSET: Y-offset from the left top corner in pixel. 
        - FG_FRAMESPERSEC: Number of images per second.
        - FG_EXPOSURE: Exposure time in µs.
        - FG_FORMAT: Color format of the transferred image
                        -# 8bit gray (FG_GRAY)
                        -# 16bit color (FG_GRAY16)
                        -# 24bit color (FG_COL24).
                     See color management of the according frame grabber design. 
        - FG_PORT: Logical number of the active CameraLink port.
        - FG_TRIGGERMODE: Trigger modes:
                            -# FREE_RUN
                            -# GRABBER_CONTROLLED
                            -# GRABBER_CONTROLLED_SYNCRON
                            -# ASYNC_SOFTWARE_TRIGGER
                            -# ASYNC_TRIGGER. 
        - FG_STROBPULSEDELAY: Strobe delay to the trigger in µs.
        - FG_GLOBAL_ACCESS: Returns the value for the set plausibility access.
        
 * \param value  Pointer to required value.
 * \return FG_OK if the parameter was read correctly
           FG_INVALID_PARAMETER if an invalid parameter has been entered.
           FG_VALUE_OUT_OF_RANGE The entered value is besides valid ranges.
 */
bool CaptureCAM_CL::setProperty( int property_id, int value )
{

    int mvret = FG_OK;

    switch(property_id)
    {
    case FG_PORT                : Fg_setParameter( fg, FG_PORT, &value, camPort); camPort = value; break;
    case FG_WIDTH               : Fg_setParameter( fg, FG_WIDTH, &value, camPort); width = value; break;
    case FG_HEIGHT              : Fg_setParameter( fg, FG_HEIGHT, &value, camPort); height = value; break;
    case FG_XOFFSET             : Fg_setParameter( fg, FG_XOFFSET, &value, camPort); xoffset = value; break;
    case FG_YOFFSET             : Fg_setParameter( fg, FG_YOFFSET, &value, camPort); yoffset = value; break;
    case FG_XSHIFT              : Fg_setParameter( fg, FG_XSHIFT, &value, camPort); break;
    case FG_TIMEOUT             : Fg_setParameter( fg, FG_TIMEOUT, &value, camPort); timeout = value; break;
    case FG_FRAMESPERSEC        : {double val = double(value); Fg_setParameter( fg, FG_FRAMESPERSEC, &val, camPort); framespersec = value; break;}
    case FG_FORMAT              : Fg_setParameter( fg, FG_FORMAT, &value, camPort); format = value; break;
    case FG_EXPOSURE            : Fg_setParameter( fg, FG_EXPOSURE, &value, camPort); exposure = value; break;
    case FG_TRIGGERMODE         : Fg_setParameter( fg, FG_TRIGGERMODE, &value, camPort); break;
    case FG_STROBEPULSEDELAY    : Fg_setParameter( fg, FG_STROBEPULSEDELAY, &value, camPort); break;
    case FG_GLOBAL_ACCESS       : Fg_setParameter( fg, FG_GLOBAL_ACCESS, &value, camPort); break;
    }

    if(mvret != FG_OK)
    {
        errMsg("Set parameter error", mvret);
        return false;
    }
    else
        return true;

}


/**
 * \fn int CaptureCAM_CL::startAcquire()
 * \brief   This function start a continuous grabbing. 
 *          Having started, an infinite number of image will be grabbed. 
 *          By default, the maximum image number is set to GRAB_INFINITE (nrOfPicturesToGrab)
 *          If a timeout occurs, the grabbing will be stopped.
 *          To manually stop the grabbing, use CaptureCAM_CL::close().
 * \return FG_OK if the acquisition has started, FG_ERROR if it has failed.
 */
int CaptureCAM_CL::startAcquire(){
    if ((Fg_AcquireEx(fg, camPort, nrOfPicturesToGrab, ACQ_STANDARD, memHandle)) < 0) {
      fprintf(stderr, "Fg_AcquireEx() failed: %s\n", Fg_getLastErrorDescription(fg));
      cout << "Start acquisition failed: " << Fg_getLastErrorDescription(fg) << endl;
      Fg_FreeMemEx(fg, memHandle);
      Fg_FreeGrabber(fg);
      return FG_ERROR;
    }
    isopened=true;
    try{
	int bitAlignment = FG_LEFT_ALIGNED;
	CHECK(FG_BITALIGNMENT, "FG_BITALIGNMENT", &bitAlignment);
	CHECK(FG_TIMEOUT, "FG_TIMEOUT", &timeout);
    }catch(string const& error){
	cout << "ERROR: " << error <<endl;
	close();
	init();
	return FG_ERROR;
    }catch(int e){
	close();
	init();
	return FG_ERROR;
    }
    return FG_OK;
}


/**
 * \fn int CaptureCAM_CL::restart()
 * \brief This function restart the device, and re-init the serial port communication.
 * \return FG_OK if the acquisition ha restarted, FG_ERROR if it has failed.
 */
int CaptureCAM_CL::restart(){
  if ((Fg_AcquireEx(fg, camPort, nrOfPicturesToGrab, ACQ_STANDARD, memHandle)) < 0) {
    fprintf(stderr, "Fg_AcquireEx() failed: %s\n", Fg_getLastErrorDescription(fg));
    Fg_FreeMemEx(fg, memHandle);
    Fg_FreeGrabber(fg);
    return FG_ERROR;
  }
//   serialInit(ComNr);
  return FG_OK;
}

void CaptureCAM_CL::errMsg(const char* msg, int errNum)
{
#if defined WIN32 || defined _WIN32
    char buf[512]="";
    sprintf( buf, "%s : %d\n", msg, errNum);
    OutputDebugString(buf);
#else
    cout << msg << errNum << endl;
#endif
}   

/** @} */ 
/** @} */
/** @} */