# coding: utf-8

from crappy.tool.ft232h import FT232H

"""
Utility for writing a serial number to an FT32H.

Simply run this program, it will ask you for the serial number to write and 
will then write it.
"""

serial_number = None

while not isinstance(serial_number, int):
  try:
    serial_number = int(input("Please enter the desired serial number : "))
  except ValueError:
    print("Wrong input, serial_number should be an integer !")
    print("")

FT232H = FT232H('Write_serial_nr', str(serial_number))

print("Serial number successfully written")
