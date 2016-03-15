from crappy.sensor import clModule as cl

class ClSerial(object):
    
    def __init__(self, obj):
	super(ClSerial, self).__init__()
	self.width_code = obj.width_code
	self.height_code = obj.height_code
	self.offsetX_code = obj.offsetX_code
	self.exposure_code = obj.exposure_code
	self.offsetY_code = obj.offsetY_code
    
    
    def getCode(self, property_id, value):
        return {
        cl.FG_WIDTH: self.width_code.format(int(value)), #self.cam.serialWrite(self.width_code.format(value)),
        cl.FG_HEIGHT: self.height_code.format(value),
        cl.FG_XOFFSET: self.offsetX_code.format(value),
        cl.FG_YOFFSET: self.offsetY_code.format(value),
        cl.FG_EXPOSURE: self.exposure_code.format(value),
        }[property_id]
        
    #def getCode(property_id, value):
        
        #if(property_id==cl.FG_WIDTH): 
	  #return self.cam.serialWrite(self.width_code.format(value))
        #if(property_id==cl.FG_HEIGHT): 
	  #return self.cam.serialWrite(self.height_code.format(value))
        #if(property_id==cl.FG_XOFFSET): 
	  #return self.cam.serialWrite(self.offsetx_code.format(value))
        #if(property_id==cl.FG_YOFFSET): 
	  #return self.cam.serialWrite(self.offsety_code.format(value))
        #if(property_id==cl.FG_EXPOSURE): 
	  #return self.cam.serialWrite(self.exposure_code.format(value))
        