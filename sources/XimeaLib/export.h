// PyObject *myarray;
map< string, int > my_map= 
{
	{"CAP_PROP_XI_DOWNSAMPLING", 400},// Change image resolution by binning or skipping.
	{"CAP_PROP_XI_DATA_FORMAT",  401},// Output data format.
	{"CAP_PROP_XI_OFFSET_X", 402},// Horizontal offset from the origin to the area of interest (in pixels).
	{"CAP_PROP_XI_OFFSET_Y" , 403},// Vertical offset from the origin to the area of interest (in pixels).
	{"CAP_PROP_XI_TRG_SOURCE", 404},// Defines source of trigger.
	{"CAP_PROP_XI_TRG_SOFTWARE", 405},// Generates an internal trigger. PRM_TRG_SOURCE must be set to TRG_SOFTWARE.
	{"CAP_PROP_XI_GPI_SELECTOR", 406},// Selects general purpose input
	{"CAP_PROP_XI_GPI_MODE", 407},// Set general purpose input mode
	{"CAP_PROP_XI_GPI_LEVEL", 408},// Get general purpose level
	{"CAP_PROP_XI_GPO_SELECTOR", 409},// Selects general purpose output
	{"CAP_PROP_XI_GPO_MODE", 410},// Set general purpose output mode
	{"CAP_PROP_XI_LED_SELECTOR", 411},// Selects camera signalling LED
	{"CAP_PROP_XI_LED_MODE", 412},// Define camera signalling LED functionality
	{"CAP_PROP_XI_MANUAL_WB", 413},// Calculates White Balance(must be called during acquisition)
	{"CAP_PROP_XI_AUTO_WB", 414},// Automatic white balance
	{"CAP_PROP_XI_AEAG", 415},// Automatic exposure/gain
	{"CAP_PROP_XI_EXP_PRIORITY", 416},// Exposure priority (0.5 - exposure 50%, gain 50%).
	{"CAP_PROP_XI_AE_MAX_LIMIT", 417},// Maximum limit of exposure in AEAG procedure
	{"CAP_PROP_XI_AG_MAX_LIMIT", 418},// Maximum limit of gain in AEAG procedure
	{"CAP_PROP_XI_AEAG_LEVEL", 419},// Average intensity of output signal AEAG should achieve(in %)
	{"CAP_PROP_XI_TIMEOUT", 420},// Image capture timeout in milliseconds
	{"CAP_PROP_XI_TIMESTAMP", 421},   // Time the image has been taken in second accurate at microsecond
	{"CAP_PROP_XI_FRAME_NUMBER", 422},// Frame number (reset by exposure, gain, downsampling change, auto exposure (AEAG)) 
	{"CAP_PROP_XI_OUTPUT_DATA_BIT_DEPTH", 423},// Number of byte of the camera (mandatory for data packing)
	{"CAP_PROP_XI_DATA_PACKING", 424},// Data packing allow to transfert efficiently image with depth over 8 bits
	{"CAP_PROP_GAIN",14},
	{"CAP_PROP_EXPOSURE",15},
	{"CAP_PROP_POS_FRAMES",1},
	{"CAP_PROP_FRAME_WIDTH",3},
	{"CAP_PROP_FRAME_HEIGHT",4},
	{"CAP_PROP_FPS",5},
	{"CAP_ANY", 0}, // autodetect
	{"CAP_VFW", 200}, // platform native
	{"CAP_V4L", 200},
	{"CAP_V4L2", CAP_V4L},
	{"CAP_FIREWARE", 300},// IEEE 1394 drivers
	{"CAP_FIREWIRE", CAP_FIREWARE},
	{"CAP_IEEE1394", CAP_FIREWARE},
	{"CAP_DC1394", CAP_FIREWARE},
	{"CAP_CMU1394", CAP_FIREWARE},
	{"CAP_QT", 500},// QuickTime
	{"CAP_UNICAP", 600},  // Unicap drivers
	{"CAP_DSHOW", 700}, // DirectShow (via videoInput)
	{"CAP_PVAPI", 800}, // PvAPI, Prosilica GigE SDK
	{"CAP_OPENNI", 900},   // OpenNI (for Kinect)
	{"CAP_OPENNI_ASUS", 910},   // OpenNI (for Asus Xtion)
	{"CAP_ANDROID", 1000},  // Android
	{"CAP_XIAPI", 1100},// XIMEA Camera API
	{"CAP_AVFOUNDATION", 1200},  // AVFoundation framework for iOS (OS X Lion will have the same API)
	{"CAP_GIGANETIX", 1300},  // Smartek Giganetix GigEVisionSDK
	{"CAP_MSMF", 1400},  // Microsoft Media Foundation (via videoInput)
	{"CAP_INTELPERC", 1500},  // Intel Perceptual Computing SDK
	{"CAP_OPENNI2", 1600 }  // OpenNI2 (for Kinect) 
};

