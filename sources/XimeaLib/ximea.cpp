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
        
        mvret = xiSetParamInt(hmv, XI_PRM_ACQ_TIMING_MODE, XI_ACQ_TIMING_MODE_FREE_RUN);
        HandleResult(mvret, "Error while setting timing mode.");
        
    }else{
        stat = xiSetParamInt(hmv, XI_PRM_BUFFERS_QUEUE_SIZE, 4);
        HandleResult(stat,"xiSetParam (XI_PRM_BUFFERS_QUEUE_SIZE)");
        stat = xiSetParamInt(hmv, XI_PRM_RECENT_FRAME, 1);
        HandleResult(stat,"xiSetParam (recent frame)");
        stat = xiSetParamInt(hmv, XI_PRM_TRG_SOURCE, XI_TRG_SOFTWARE);
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
//     cout << "Number of connected devices: " << numDevices << endl;
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
bool CaptureCAM_XIMEA::open( char * device_path )
{
    cout << "OPEN DEVICE BY NAME" << endl;
    #define HandleXiResult(res) if (res!=XI_OK)  goto error;
    int mvret = XI_OK;

    if(numDevices == 0)
        return false;
    
    if(xiOpenDeviceBy(XI_OPEN_BY_INST_PATH, device_path, &hmv) != XI_OK)
    {
#if defined WIN32 || defined _WIN32
        errMsg("Open XI_DEVICE failed", mvret);
        return false;
#else
        // try opening second time if first fails
        if(xiOpenDeviceBy(XI_OPEN_BY_INST_PATH, device_path, &hmv)  != XI_OK)
        {
            errMsg("Open XI_DEVICE failed", mvret);
            return false;
        }
#endif
    }
    int width   = 0;
    int height  = 0;
    int isColor = 0;

    // always use auto exposure/gain
    mvret = xiSetParamInt( hmv, XI_PRM_AEAG, 1);
    HandleXiResult(mvret);

    mvret = xiGetParamInt( hmv, XI_PRM_WIDTH, &width);
    HandleXiResult(mvret);

    mvret = xiGetParamInt( hmv, XI_PRM_HEIGHT, &height);
    HandleXiResult(mvret);

    mvret = xiGetParamInt(hmv, XI_PRM_IMAGE_IS_COLOR, &isColor);
    HandleXiResult(mvret);
    
    mvret = xiSetParamInt(hmv, XI_PRM_ACQ_TIMING_MODE, XI_ACQ_TIMING_MODE_FREE_RUN);
    HandleXiResult(mvret);
    
    if(isColor) // for color cameras
    {
        // default image format RGB24
        mvret = xiSetParamInt( hmv, XI_PRM_IMAGE_DATA_FORMAT, XI_RGB24);
        HandleXiResult(mvret);

        // always use auto white balance for color cameras
        mvret = xiSetParamInt( hmv, XI_PRM_AUTO_WB, 1);
        HandleXiResult(mvret);
    }
    else // for mono cameras
    {
        // default image format MONO8
        mvret = xiSetParamInt( hmv, XI_PRM_IMAGE_DATA_FORMAT, XI_MONO8);
        HandleXiResult(mvret);
    }

    //default capture timeout 10s
    timeout = 10000;

    stat = xiSetParamInt(hmv, XI_PRM_BUFFERS_QUEUE_SIZE, 4);
    HandleResult(stat,"xiSetParam (XI_PRM_BUFFERS_QUEUE_SIZE)");
    stat = xiSetParamInt(hmv, XI_PRM_RECENT_FRAME, 1);
    HandleResult(stat,"xiSetParam (recent frame)");
    stat = xiSetParamInt(hmv, XI_PRM_TRG_SOURCE, XI_TRG_SOFTWARE);
    HandleResult(mvret, "Error while disabling external trigger source");

    mvret = xiStartAcquisition(hmv);
    if(mvret != XI_OK)
    {
        errMsg("StartAcquisition XI_DEVICE failed", mvret);
        goto error;
    }
    xiSetParamInt(hmv, XI_PRM_TS_RST_MODE , XI_TS_RST_ARM_ONCE);
	xiSetParamInt(hmv, XI_PRM_TS_RST_SOURCE , XI_TS_RST_SRC_SW);

    isopened=true;
    return true;

error:
    errMsg("Open XI_DEVICE failed", mvret);
    xiCloseDevice(hmv);
    hmv = NULL;
    return false;
}

// Initialize camera input
bool CaptureCAM_XIMEA::open( int wIndex )
{
    #define HandleXiResult(res) if (res!=XI_OK)  goto error;
    cout << "OPEN DEVICE BY ID" << endl;
    int mvret = XI_OK;

    if(numDevices == 0)
        return false;
    
    
    if((mvret = xiOpenDevice( wIndex, &hmv)) != XI_OK)
    {
#if defined WIN32 || defined _WIN32
        errMsg("Open XI_DEVICE failed", mvret);
        return false;
#else
        // try opening second time if first fails
        if((mvret = xiOpenDevice( wIndex, &hmv))  != XI_OK)
        {
            errMsg("Open XI_DEVICE failed", mvret);
            return false;
        }
#endif
    }

    int width   = 0;
    int height  = 0;
    int isColor = 0;
    
    // always use auto exposure/gain
    mvret = xiSetParamInt(hmv, XI_PRM_AEAG, 1);
    HandleXiResult(mvret);

    mvret = xiGetParamInt(hmv, XI_PRM_WIDTH, &width);
    HandleXiResult(mvret);

    mvret = xiGetParamInt(hmv, XI_PRM_HEIGHT, &height);
    HandleXiResult(mvret);

    mvret = xiGetParamInt(hmv, XI_PRM_IMAGE_IS_COLOR, &isColor);
    HandleXiResult(mvret);

    if(isColor) // for color cameras
    {
        // default image format RGB24
        mvret = xiSetParamInt( hmv, XI_PRM_IMAGE_DATA_FORMAT, XI_RGB24);
        HandleXiResult(mvret);

        // always use auto white balance for color cameras
        mvret = xiSetParamInt( hmv, XI_PRM_AUTO_WB, 1);
        HandleXiResult(mvret);
    }
    else // for mono cameras
    {
        // default image format MONO8
        mvret = xiSetParamInt( hmv, XI_PRM_IMAGE_DATA_FORMAT, XI_MONO8);
        HandleXiResult(mvret);
    }

    //default capture timeout 10s
    timeout = 10000;
    
    /* NOT WORKING IF XI_PRM_BUFFERS_QUEUE_SIZE is set to a value < 3*/
//    mvret = xiSetParamInt(hmv, XI_PRM_BUFFERS_QUEUE_SIZE, 3);
//    if( mvret != XI_OK)
//        errMsg("Set parameter error", mvret);
//    mvret = xiSetParamInt(hmv, XI_PRM_RECENT_FRAME, 1);
//    if( mvret != XI_OK)
//        errMsg("Set parameter error", mvret);
//    mvret = xiSetParamInt(hmv, XI_PRM_TRG_SOURCE, XI_TRG_SOFTWARE);
//    if( mvret != XI_OK){
//        errMsg("Error while disabling external trigger source", mvret);
//    }
//
//    stat = xiSetParamInt(hmv, XI_PRM_BUFFERS_QUEUE_SIZE, 4);
//    HandleResult(stat,"xiSetParam (XI_PRM_BUFFERS_QUEUE_SIZE)");
//    stat = xiSetParamInt(hmv, XI_PRM_RECENT_FRAME, 1);
//    HandleResult(stat,"xiSetParam (recent frame)");

    mvret = xiStartAcquisition(hmv);
    if(mvret != XI_OK)
    {
        errMsg("StartAcquisition XI_DEVICE failed", mvret);
        goto error;
    }
    xiSetParamInt(hmv, XI_PRM_TS_RST_MODE , XI_TS_RST_ARM_ONCE);
	xiSetParamInt(hmv, XI_PRM_TS_RST_SOURCE , XI_TS_RST_SRC_SW);
    
    isopened=true;
    return true;

error:
    errMsg("Open XI_DEVICE failed", mvret);
    xiCloseDevice(hmv);
    hmv = NULL;
    return false;
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
    // image.width = width;
    // image.height = height;
    // image.AbsoluteOffsetX= xoffset;
    // image.AbsoluteOffsetY= yoffset;
//    xiSetParamInt(hmv, XI_PRM_TRG_SOFTWARE, 1);
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
    case CAP_PROP_FPS          : xiGetParamFloat( hmv, XI_PRM_FRAMERATE, &fval); cout << fval << endl; return fval;
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
    // with XI_OK there is nothing to report
    if(errNum == XI_OK) return;
    string error_message = "";
    switch(errNum)
    {

    case XI_OK : error_message = "Function call succeeded"; break;
    case XI_INVALID_HANDLE : error_message = "Invalid handle"; break;
    case XI_READREG : error_message = "Register read error"; break;
    case XI_WRITEREG : error_message = "Register write error"; break;
    case XI_FREE_RESOURCES : error_message = "Freeing resiurces error"; break;
    case XI_FREE_CHANNEL : error_message = "Freeing channel error"; break;
    case XI_FREE_BANDWIDTH : error_message = "Freeing bandwith error"; break;
    case XI_READBLK : error_message = "Read block error"; break;
    case XI_WRITEBLK : error_message = "Write block error"; break;
    case XI_NO_IMAGE : error_message = "No image"; break;
    case XI_TIMEOUT : error_message = "Timeout"; break;
    case XI_INVALID_ARG : error_message = "Invalid arguments supplied"; break;
    case XI_NOT_SUPPORTED : error_message = "Not supported"; break;
    case XI_ISOCH_ATTACH_BUFFERS : error_message = "Attach buffers error"; break;
    case XI_GET_OVERLAPPED_RESULT : error_message = "Overlapped result"; break;
    case XI_MEMORY_ALLOCATION : error_message = "Memory allocation error"; break;
    case XI_DLLCONTEXTISNULL : error_message = "DLL context is NULL"; break;
    case XI_DLLCONTEXTISNONZERO : error_message = "DLL context is non zero"; break;
    case XI_DLLCONTEXTEXIST : error_message = "DLL context exists"; break;
    case XI_TOOMANYDEVICES : error_message = "Too many devices connected"; break;
    case XI_ERRORCAMCONTEXT : error_message = "Camera context error"; break;
    case XI_UNKNOWN_HARDWARE : error_message = "Unknown hardware"; break;
    case XI_INVALID_TM_FILE : error_message = "Invalid TM file"; break;
    case XI_INVALID_TM_TAG : error_message = "Invalid TM tag"; break;
    case XI_INCOMPLETE_TM : error_message = "Incomplete TM"; break;
    case XI_BUS_RESET_FAILED : error_message = "Bus reset error"; break;
    case XI_NOT_IMPLEMENTED : error_message = "Not implemented"; break;
    case XI_SHADING_TOOBRIGHT : error_message = "Shading too bright"; break;
    case XI_SHADING_TOODARK : error_message = "Shading too dark"; break;
    case XI_TOO_LOW_GAIN : error_message = "Gain is too low"; break;
    case XI_INVALID_BPL : error_message = "Invalid bad pixel list"; break;
    case XI_BPL_REALLOC : error_message = "Bad pixel list realloc error"; break;
    case XI_INVALID_PIXEL_LIST : error_message = "Invalid pixel list"; break;
    case XI_INVALID_FFS : error_message = "Invalid Flash File System"; break;
    case XI_INVALID_PROFILE : error_message = "Invalid profile"; break;
    case XI_INVALID_CALIBRATION : error_message = "Invalid calibration"; break;
    case XI_INVALID_BUFFER : error_message = "Invalid buffer"; break;
    case XI_INVALID_DATA : error_message = "Invalid data"; break;
    case XI_TGBUSY : error_message = "Timing generator is busy"; break;
    case XI_IO_WRONG : error_message = "Wrong operation open/write/read/close"; break;
    case XI_ACQUISITION_ALREADY_UP : error_message = "Acquisition already started"; break;
    case XI_OLD_DRIVER_VERSION : error_message = "Old version of device driver installed to the system."; break;
    case XI_GET_LAST_ERROR : error_message = "To get error code please call GetLastError function."; break;
    case XI_CANT_PROCESS : error_message = "Data cant be processed"; break;
    case XI_ACQUISITION_STOPED : error_message = "Acquisition has been stopped. It should be started before GetImage."; break;
    case XI_ACQUISITION_STOPED_WERR : error_message = "Acquisition has been stoped with error."; break;
    case XI_INVALID_INPUT_ICC_PROFILE : error_message = "Input ICC profile missed or corrupted"; break;
    case XI_INVALID_OUTPUT_ICC_PROFILE : error_message = "Output ICC profile missed or corrupted"; break;
    case XI_DEVICE_NOT_READY : error_message = "Device not ready to operate"; break;
    case XI_SHADING_TOOCONTRAST : error_message = "Shading too contrast"; break;
    case XI_ALREADY_INITIALIZED : error_message = "Module already initialized"; break;
    case XI_NOT_ENOUGH_PRIVILEGES : error_message = "Application doesnt enough privileges(one or more app"; break;
    case XI_NOT_COMPATIBLE_DRIVER : error_message = "Installed driver not compatible with current software"; break;
    case XI_TM_INVALID_RESOURCE : error_message = "TM file was not loaded successfully from resources"; break;
    case XI_DEVICE_HAS_BEEN_RESETED : error_message = "Device has been reseted, abnormal initial state"; break;
    case XI_NO_DEVICES_FOUND : error_message = "No Devices Found"; break;
    case XI_RESOURCE_OR_FUNCTION_LOCKED : error_message = "Resource(device) or function locked by mutex"; break;
    case XI_BUFFER_SIZE_TOO_SMALL : error_message = "Buffer provided by user is too small"; break;
    case XI_COULDNT_INIT_PROCESSOR : error_message = "Couldnt initialize processor."; break;
    case XI_NOT_INITIALIZED : error_message = "The object/module/procedure/process being referred to has not been started."; break;
    case XI_UNKNOWN_PARAM : error_message = "Unknown parameter"; break;
    case XI_WRONG_PARAM_VALUE : error_message = "Wrong parameter value"; break;
    case XI_WRONG_PARAM_TYPE : error_message = "Wrong parameter type"; break;
    case XI_WRONG_PARAM_SIZE : error_message = "Wrong parameter size"; break;
    case XI_BUFFER_TOO_SMALL : error_message = "Input buffer too small"; break;
    case XI_NOT_SUPPORTED_PARAM : error_message = "Parameter info not supported"; break;
    case XI_NOT_SUPPORTED_PARAM_INFO : error_message = "Parameter info not supported"; break;
    case XI_NOT_SUPPORTED_DATA_FORMAT : error_message = "Data format not supported"; break;
    case XI_READ_ONLY_PARAM : error_message = "Read only parameter"; break;
    case XI_BANDWIDTH_NOT_SUPPORTED : error_message = "This camera does not support currently available bandwidth"; break;
    case XI_INVALID_FFS_FILE_NAME : error_message = "FFS file selector is invalid or NULL"; break;
    case XI_FFS_FILE_NOT_FOUND : error_message = "FFS file not found"; break;
    case XI_PROC_OTHER_ERROR : error_message = "Processing error - other"; break;
    case XI_PROC_PROCESSING_ERROR : error_message = "Error while image processing."; break;
    case XI_PROC_INPUT_FORMAT_UNSUPPORTED : error_message = "Input format is not supported for processing."; break;
    case XI_PROC_OUTPUT_FORMAT_UNSUPPORTED : error_message = "Output format is not supported for processing."; break;
    default:
        error_message = "Unknown error value";
    }

    #if defined WIN32 || defined _WIN32
    char buf[512]="";
    sprintf( buf, "%s : %d, %s\n", msg, errNum, error_message.c_str());
    OutputDebugString(buf);
    #else
    fprintf(stderr, "%s : %d, %s\n", msg, errNum, error_message.c_str());
    #endif
}

/** @} */ 
/** @} */ 