#include <map>
map< string, int > my_map= 
{
	{"FG_OK", FG_OK},// The parameter was set correctly.
	{"FG_NOT_INIT ",  FG_NOT_INIT},// Initialization failed.
	{"FG_INVALID_PARAMETER", FG_INVALID_PARAMETER},// An invalid parameter has been entered.
	{"FG_VALUE_OUT_OF_RANGE" , FG_VALUE_OUT_OF_RANGE},// Value is besides valid ranges.
	{"FG_CAMSTATUS", FG_CAMSTATUS},// If a camera signal is on CameraLink port value is 1 else 0.
	{"FG_REVNR ", FG_REVNR},// Current revision version of camera DLL.
	{"FG_TIMEOUT", FG_TIMEOUT},// Time in seconds until device driver displays a timeout of the frame grabber.
	{"FG_WIDTH", FG_WIDTH},//  Width of the clipping image.
	{"FG_MAXWIDTH", FG_MAXWIDTH},// Maximum width of the clipping image.
	{"FG_HEIGHT", FG_HEIGHT},// Height of the clipping image.
	{"FG_MAXHEIGHT", FG_MAXHEIGHT},// Maximum height of the clipping image.
	{"FG_XSHIFT", FG_XSHIFT},// Number of invalid words at the beginning of a row (modulo of the width of the interface). 
	{"FG_XOFFSET", FG_XOFFSET},// X-offset from the left top corner in pixel. 
	{"FG_YOFFSET", FG_YOFFSET},// Y-offset from the left top corner in pixel. 
	{"FG_FRAMESPERSEC", FG_FRAMESPERSEC},// Number of images per second.
	{"FG_EXPOSURE", FG_EXPOSURE},// Exposure time in µs.
        /*
         * Color format of the transferred image: 
         *  1bit (FG_BINARY),
         *  8bit (FG_GRAY),
         *  16bit (FG_GRAY16),
         *  24bit (FG_COL24),
         *  30bit (FG_COL30),
         *  32bit (FG_COL32),
         *  48bit (FG_COL48).
         * See color management of the according frame grabber design.  
         */
        {"FG_FORMAT", FG_FORMAT},
        
        {"FG_BINARY", FG_BINARY},
        {"FG_GRAY", FG_GRAY},
	{"FG_GRAY16 ", FG_GRAY16},
	{"FG_COL24", FG_COL24},
	{"FG_COL30", FG_COL30},
        {"FG_COL32", FG_COL32},
	{"FG_COL48 ", FG_COL48},
	
        {"PORT_A", PORT_A}, // base config
        {"PORT_B", PORT_B}, // base config
	{"FG_COL48 ", PORT_AB}, // medium/full config
        
	{"FG_PORT", FG_PORT},//Logical number of the active CameraLink port.
        {"FG_PIXELDEPTH", FG_PIXELDEPTH},// Returns the depth of color of the pixel. 
	{"FG_LINEALIGNMENT ", FG_LINEALIGNMENT},// Returns the alignment of a line (in bits).
	{"FG_RIGHT_ALIGNED", FG_RIGHT_ALIGNED}, // right
	{"FG_LEFT_ALIGNED ", FG_LEFT_ALIGNED}, // left
	
	{"FG_TRANSFER_LEN", FG_TRANSFER_LEN},// Returns the length of the last DMA transfer.
	
        /* 
         *Trigger modes: 
         *  - FREE_RUN
         *  - GRABBER_CONTROLLED
         *  - GRABBER_CONTROLLED_SYNCRON
         *  - ASYNC_SOFTWARE_TRIGGER
         *  - ASYNC_TRIGGER 
         */
	{"FG_TRIGGERMODE", FG_TRIGGERMODE},
        
	{"FREE_RUN", FREE_RUN},
	{"GRABBER_CONTROLLED",  GRABBER_CONTROLLED},
// 	{"GRABBER_CONTROLLED_SYNCRON", GRABBER_CONTROLLED_SYNCRON},
	{"ASYNC_SOFTWARE_TRIGGER" , ASYNC_SOFTWARE_TRIGGER},
	{"ASYNC_TRIGGER" , ASYNC_TRIGGER},
	
// 	{"FG_STROBPULSEDELAY", FG_STROBPULSEDELAY},// Strobe delay to the trigger in µs.
	{"FG_TWOCAMMODEL", FG_TWOCAMMODEL},// Returns the value, if the loaded camera applet is a dual (1) or a single applet (0).
	{"FG_HDSYNC", FG_HDSYNC},// Returns the HDSYNC value. 
	{"FG_GLOBAL_ACCESS", FG_GLOBAL_ACCESS},// Returns the value for the set plausibility access.
	 
        /*
         * Information on the board type: 
         *
         *  BINFO_BOARD_TYPE:
         *      0xa40 for microEnable IV-Base x1
         *      0xa41 for microEnable IV-Full x1
         *      0xa44 for microEnable IV-Full x4
         *
         *  BINFO_POCL:
         *      0 for microEnable IV-Base x1
         *      1 for microEnable IV-Base x1 PoCL 
         */
        {"FG_BOARD_INFORMATION", FG_BOARD_INFORMATION}
};

