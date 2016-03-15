/** @addtogroup sources
 *  
 *  More documentation for the first group.
 *  @{
 */

/** @defgroup ximea Ximea
 *  Ximea module to be interfaced with Python.
 *  @{
 */
#include "ximea.h"

CaptureCAM_XIMEA::CaptureCAM_XIMEA() {
	isopened=false;
	init();
}

CaptureCAM_XIMEA::~CaptureCAM_XIMEA(){
	close();
}


void CaptureCAM_XIMEA::addTrigger(int timout, bool triggered)
{
    int mvret = XI_OK;
    mvret = xiStopAcquisition(hmv);
    HandleResult(mvret, "Acquisition stopped");
    isopened=false;
    if(triggered){
        // select trigger source
        mvret = xiSetParamInt(hmv, XI_PRM_TRG_SOURCE, XI_TRG_EDGE_RISING);
        HandleResult(mvret, "Error while activating external trigger source");
        // select input pin 1 mode
        mvret = xiSetParamInt(hmv, XI_PRM_GPI_SELECTOR, 1);
        HandleResult(mvret, "Error while setting input pin");
        mvret = xiSetParamInt(hmv, XI_PRM_GPI_MODE, XI_GPI_TRIGGER);
        HandleResult(mvret, "Error while setting input pin mode");
        // set digital output 1 mode
        mvret = xiSetParamInt(hmv, XI_PRM_GPO_SELECTOR, 1);
        HandleResult(mvret, "Error while setting digital ouput");
        mvret = xiSetParamInt(hmv, XI_PRM_GPO_MODE,  XI_GPO_EXPOSURE_ACTIVE);
        HandleResult(mvret, "Error while setting digital output mode");
    }else{
        mvret = xiSetParamInt(hmv, XI_PRM_TRG_SOURCE,  XI_TRG_OFF);
        HandleResult(mvret, "Error while disabling external trigger source");
    }
    mvret = xiStartAcquisition(hmv);
    if(mvret != XI_OK)
    {
        errMsg("StartAcquisition XI_DEVICE failed", mvret);
        close();
    }
    timeout = timout;
    isopened=true;
}

void CaptureCAM_XIMEA::init()
{
    stat = xiGetNumberDevices(&numDevices);
    HandleResult(stat,"xiGetNumberDevices (no camera found)");
    try{
        if (!numDevices)
        {
            throw "No camera found\n";
        }
        hmv = NULL;
        isopened=false;
        timeout = 0;
        memset(&image, 0, sizeof(XI_IMG));
    }catch(const char* e){
        cout<< "An Exception occured. Exception: "<<e<<endl;
        close();
    }
}


// Initialize camera input
bool CaptureCAM_XIMEA::open( int wIndex )
{
    int       isColor;
    int mvret = XI_OK;
    if(numDevices == 0)
        return false;
    stat = xiOpenDevice(wIndex, &hmv);
    HandleResult(stat,"Open XI_DEVICE failed");
    stat = xiSetParamInt(hmv, XI_PRM_EXPOSURE, 10000);
    HandleResult(stat,"xiSetParam (exposure set)");
    // always use auto exposure/gain
    mvret = xiSetParamInt( hmv, XI_PRM_AEAG, 1);
    HandleResult(mvret, "error while setting exposure and gain auto\n");
    mvret = xiGetParamInt( hmv, XI_PRM_WIDTH, &width);
    HandleResult(mvret, "error while setting width");
    mvret = xiGetParamInt( hmv, XI_PRM_HEIGHT, &height);
    HandleResult(mvret, "error while setting height");
    mvret = xiGetParamInt(hmv, XI_PRM_IMAGE_IS_COLOR, &isColor);
    HandleResult(mvret, "error while setting color parameter ");
    mvret = xiGetParamInt(hmv,  XI_PRM_IMAGE_DATA_FORMAT, &format);
    xiSetParamInt( hmv, XI_PRM_DOWNSAMPLING_TYPE, 1);
    xiSetParamInt(hmv, XI_PRM_IMAGE_DATA_FORMAT, XI_MONO8);
    HandleResult(mvret, "error while setting data format");
    
    timeout = 10000;
    mvret = xiStartAcquisition(hmv);
    if(mvret != XI_OK)
    {
        errMsg("StartAcquisition XI_DEVICE failed", mvret);
        close();
    }
	
    isopened=true;
    return true;
}

void CaptureCAM_XIMEA::close()
{
    if(hmv)
    {
        xiStopAcquisition(hmv);
        xiCloseDevice(hmv);
        isopened=false;
    }
    hmv = NULL;
}

bool CaptureCAM_XIMEA::grabFrame()
{
    memset(&image, 0, sizeof(XI_IMG));
    image.size = sizeof(XI_IMG);
    image.width = width;
    image.height = height;
    image.AbsoluteOffsetX= xoffset;
    image.AbsoluteOffsetY= yoffset;
    int stat = xiGetImage( hmv, timeout, &image);
    if(stat == MM40_ACQUISITION_STOPED)
    {
        xiStartAcquisition(hmv);
        stat = xiGetImage(hmv, timeout, &image);
    }
    if(stat != XI_OK)
    {
        errMsg("Error during GetImage", stat);
        return false;
    }
    return true;
}

void CaptureCAM_XIMEA::resetCvImage()
{
    if( (int)image.width != width || (int)image.height != height || image.frm != (XI_IMG_FORMAT)format)
    {
	xiGetParamInt( hmv, XI_PRM_WIDTH, &width);
	xiGetParamInt( hmv, XI_PRM_HEIGHT, &height);
	xiGetParamInt( hmv, XI_PRM_IMAGE_DATA_FORMAT, &format);
    }
}

double CaptureCAM_XIMEA::getProperty( int property_id )
{
    if(hmv == NULL)
        return 0;

    int ival = 0;
    float fval = 0;

    switch( property_id )
    {
    // OCV parameters
    case CAP_PROP_POS_FRAMES   : return (double) image.nframe;
    case CAP_PROP_FRAME_WIDTH  : xiGetParamInt( hmv, XI_PRM_WIDTH, &ival); return ival;
    case CAP_PROP_FRAME_HEIGHT : xiGetParamInt( hmv, XI_PRM_HEIGHT, &ival); return ival;
    case CAP_PROP_FPS          : xiGetParamFloat( hmv, XI_PRM_FRAMERATE, &fval); return fval;
    case CAP_PROP_GAIN         : xiGetParamFloat( hmv, XI_PRM_GAIN, &fval); return fval;
    case CAP_PROP_EXPOSURE     : xiGetParamInt( hmv, XI_PRM_EXPOSURE, &ival); return ival;

    // XIMEA camera properties
    case CAP_PROP_XI_DOWNSAMPLING  : xiGetParamInt( hmv, XI_PRM_DOWNSAMPLING, &ival); return ival;
    case CAP_PROP_XI_DATA_FORMAT   : xiGetParamInt( hmv, XI_PRM_IMAGE_DATA_FORMAT, &ival);  return ival;
    case CAP_PROP_XI_OFFSET_X      : xiGetParamInt( hmv, XI_PRM_OFFSET_X, &ival); return ival;
    case CAP_PROP_XI_OFFSET_Y      : xiGetParamInt( hmv, XI_PRM_OFFSET_Y, &ival); return ival;
    case CAP_PROP_XI_TRG_SOURCE    : xiCloseDevice(hmv);xiGetParamInt( hmv, XI_PRM_TRG_SOURCE, &ival); return ival;
    case CAP_PROP_XI_GPI_SELECTOR  : xiGetParamInt( hmv, XI_PRM_GPI_SELECTOR, &ival); return ival;
    case CAP_PROP_XI_GPI_MODE      : xiGetParamInt( hmv, XI_PRM_GPI_MODE, &ival); return ival;
    case CAP_PROP_XI_GPI_LEVEL     : xiGetParamInt( hmv, XI_PRM_GPI_LEVEL, &ival); return ival;
    case CAP_PROP_XI_GPO_SELECTOR  : xiGetParamInt( hmv, XI_PRM_GPO_SELECTOR, &ival); return ival;
    case CAP_PROP_XI_GPO_MODE      : xiGetParamInt( hmv, XI_PRM_GPO_MODE, &ival); return ival;
    case CAP_PROP_XI_LED_SELECTOR  : xiGetParamInt( hmv, XI_PRM_LED_SELECTOR, &ival); return ival;
    case CAP_PROP_XI_LED_MODE      : xiGetParamInt( hmv, XI_PRM_LED_MODE, &ival); return ival;
    case CAP_PROP_XI_AUTO_WB       : xiGetParamInt( hmv, XI_PRM_AUTO_WB, &ival); return ival;
    case CAP_PROP_XI_AEAG          : xiGetParamInt( hmv, XI_PRM_AEAG, &ival); return ival;
    case CAP_PROP_XI_EXP_PRIORITY  : xiGetParamFloat( hmv, XI_PRM_EXP_PRIORITY, &fval); return fval;
    case CAP_PROP_XI_AE_MAX_LIMIT  : xiGetParamInt( hmv, XI_PRM_EXP_PRIORITY, &ival); return ival;
    case CAP_PROP_XI_AG_MAX_LIMIT  : xiGetParamFloat( hmv, XI_PRM_AG_MAX_LIMIT, &fval); return fval;
    case CAP_PROP_XI_AEAG_LEVEL    : xiGetParamInt( hmv, XI_PRM_AEAG_LEVEL, &ival); return ival;
    case CAP_PROP_XI_TIMEOUT       : return timeout;
    case CAP_PROP_XI_TIMESTAMP     : return (float)image.tsSec+(float)image.tsUSec/1000000;
    case CAP_PROP_XI_FRAME_NUMBER  : return (int)image.nframe;

    }
    return 0;
}


bool CaptureCAM_XIMEA::setProperty( int property_id, double value )
{
    int ival = (int) value;
    float fval = (float) value;

    int mvret = XI_OK;

    switch(property_id)
    {
    // OCV parameters
    case CAP_PROP_FRAME_WIDTH  : mvret = xiSetParamInt( hmv, XI_PRM_WIDTH, ival); width = ival; break;
    case CAP_PROP_FRAME_HEIGHT : mvret = xiSetParamInt( hmv, XI_PRM_HEIGHT, ival); height=ival; break;
    case CAP_PROP_FPS          : mvret = xiSetParamFloat( hmv, XI_PRM_FRAMERATE, fval); break;
    case CAP_PROP_GAIN         : mvret = xiSetParamFloat( hmv, XI_PRM_GAIN, fval); break;
    case CAP_PROP_EXPOSURE     : mvret = xiSetParamInt( hmv, XI_PRM_EXPOSURE, ival); break;
    // XIMEA camera properties
    case CAP_PROP_XI_DOWNSAMPLING  		: mvret = xiSetParamInt( hmv, XI_PRM_DOWNSAMPLING, ival); break;
    case CAP_PROP_XI_DATA_FORMAT   		: mvret = xiSetParamInt( hmv, XI_PRM_IMAGE_DATA_FORMAT, ival); break;
    case CAP_PROP_XI_OFFSET_X      		: mvret = xiSetParamInt( hmv, XI_PRM_OFFSET_X, ival); xoffset = ival; break;
    case CAP_PROP_XI_OFFSET_Y      		: mvret = xiSetParamInt( hmv, XI_PRM_OFFSET_Y, ival); yoffset = ival; break;
    case CAP_PROP_XI_TRG_SOURCE    		: xiStopAcquisition(hmv);mvret = xiSetParamInt( hmv, XI_PRM_TRG_SOURCE, ival);xiStartAcquisition(hmv); break;
    case CAP_PROP_XI_OUTPUT_DATA_BIT_DEPTH  	: xiStopAcquisition(hmv);mvret = xiSetParamInt( hmv, XI_PRM_OUTPUT_DATA_BIT_DEPTH, ival); xiStartAcquisition(hmv); break; 
    case CAP_PROP_XI_DATA_PACKING	  	: xiStopAcquisition(hmv);mvret = xiSetParamInt( hmv, XI_PRM_OUTPUT_DATA_PACKING, ival); xiStartAcquisition(hmv); break; 
    case CAP_PROP_XI_GPI_SELECTOR  		: mvret = xiSetParamInt( hmv, XI_PRM_GPI_SELECTOR, ival); break;
    case CAP_PROP_XI_TRG_SOFTWARE  		: mvret = xiSetParamInt( hmv, XI_PRM_TRG_SOURCE, 1); break;
    case CAP_PROP_XI_GPI_MODE      		: mvret = xiSetParamInt( hmv, XI_PRM_GPI_MODE, ival); break;
    case CAP_PROP_XI_GPI_LEVEL     		: mvret = xiSetParamInt( hmv, XI_PRM_GPI_LEVEL, ival); break;
    case CAP_PROP_XI_GPO_SELECTOR  		: mvret = xiSetParamInt( hmv, XI_PRM_GPO_SELECTOR, ival); break;
    case CAP_PROP_XI_GPO_MODE      		: mvret = xiSetParamInt( hmv, XI_PRM_GPO_MODE, ival); break;
    case CAP_PROP_XI_LED_SELECTOR  		: mvret = xiSetParamInt( hmv, XI_PRM_LED_SELECTOR, ival); break;
    case CAP_PROP_XI_LED_MODE      		: mvret = xiSetParamInt( hmv, XI_PRM_LED_MODE, ival); break;
    case CAP_PROP_XI_AUTO_WB       		: mvret = xiSetParamInt( hmv, XI_PRM_AUTO_WB, ival); break;
    case CAP_PROP_XI_MANUAL_WB     		: mvret = xiSetParamInt( hmv, XI_PRM_LED_MODE, ival); break;
    case CAP_PROP_XI_AEAG          		: mvret = xiSetParamInt( hmv, XI_PRM_AEAG, ival); break;
    case CAP_PROP_XI_EXP_PRIORITY  		: mvret = xiSetParamFloat( hmv, XI_PRM_EXP_PRIORITY, fval); break;
    case CAP_PROP_XI_AE_MAX_LIMIT  		: mvret = xiSetParamInt( hmv, XI_PRM_EXP_PRIORITY, ival); break;
    case CAP_PROP_XI_AG_MAX_LIMIT  		: mvret = xiSetParamFloat( hmv, XI_PRM_AG_MAX_LIMIT, fval); break;
    case CAP_PROP_XI_AEAG_LEVEL    		: mvret = xiSetParamInt( hmv, XI_PRM_AEAG_LEVEL, ival); break;
    case CAP_PROP_XI_TIMEOUT       		: timeout = ival; break;
    }

    if(mvret != XI_OK)
    {
        errMsg("Set parameter error", mvret);
        return false;
    }
    else
        return true;

}


void CaptureCAM_XIMEA::errMsg(const char* msg, int errNum)
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