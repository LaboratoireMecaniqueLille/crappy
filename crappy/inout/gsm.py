# coding: utf-8

import time
from typing import List
import serial
from .inout import InOut


class Gsm(InOut):
    """Block for sending any messages by SMS to phone numbers

    Note:
        This block have to be associated with a modifier to manage 
        which message should be sent
    """

    def __init__(self, numero:List=None, port:str="/dev/ttyUSB0", baudrate:int=115200):
        """Checks arguments validity

        Args:
            numero(:obj:`list`): A list of numbers to contact.
                ::
                    syntax : "0611223344"

            port (:obj:`str`,optional): Serial port of the GSM.

            baudrate(:obj:`int`, optional): Baudrate between 1200 and 115200.
        """
        super().__init__()
        self.ser = serial.Serial(port, baudrate)
        self.sent = True
        self.number = numero

        # Change the type of numbers to bytes rather than string
        if self.number is not None:
            for x in range(len(self.number)):
                self.number[x] = self.number[x].encode('utf-8')

    def open(self) -> None:
        """Call the ``is_connected()`` method"""
        self.is_connected()
    
    def set_cmd(self, *cmd) -> None:
        """Send SMS to all phone numbers if not equal to "" """
        if cmd[0] != "" and cmd[0] != 0:
            self.send_mess(cmd[0])
        else:
            pass

    def send_mess(self, message):
        for numero in self.number:
            data = ""
            num = 0
            self.ser.write(b'AT'+b'\r\n')
            w_buff = [b"AT+CMGF=1\r\n",
                      b"AT+CMGS=\"" + numero + b"\"\r\n", message.encode()]
            while num <= 5:
                while self.ser.inWaiting() > 0:
                    data += self.ser.read(self.ser.inWaiting()).decode()
                    # Get all the answers in Waiting
                if data != "":
                    print(data)
                    if num < 2:
                        time.sleep(1)
                        self.ser.write(w_buff[num])
                        # Put the message in text mode then enter the
                        # number to contact
                    if num == 2:
                        time.sleep(0.5)
                        self.ser.write(w_buff[2])  # Write the message
                        self.ser.write(b"\x1a\r\n")
                        # 0x1a : send   0x1b : Cancel send
                    num = num + 1
                    data = "" 

    def is_connected(self) -> None:
        """Send "AT" to the GSM and wait for an response : "OK" """
        self.ser.write(b'AT' + b'\r\n')
        data = ""
        num = 0
        while num < 2:
            while self.ser.inWaiting() > 0:
                data += self.ser.read(self.ser.inWaiting()).decode()
            if data != "":    
                print(data)
                num = num + 1
                data = ""
        
    def close(self) -> None:
        """Close the serial port"""
        if self.ser is not None:
            self.ser.close()
