class JaiSerial:
    def __init__(self):
        print 'INIT JaiSerial'
        self.exposure_code = "PE={0}(0x10)\r\n"
        self.width_code = "WTC={0}(0x10)\r\n"
        self.height_code = "HTL={0}(0x10)\r\n"
        self.offsetX_code = "OFC={0}(0x10)\r\n"
        self.offsetY_code = "OFL={0}(0x10)\r\n"
