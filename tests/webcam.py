#coding: utf-8

import unittest
import cv2

class TestWebcam(unittest.TestCase):
  def test_open_device(self):
    cap = cv2.VideoCapture(0)
    self.assertTrue(cap.isOpened())

  def test_close_device(self):
    cap = cv2.VideoCapture(0)
    cap.release()
    self.assertFalse(cap.isOpened())

  def test_read(self):
    cap = cv2.VideoCapture(0)
    r,frame = cap.read()
    self.assertTrue(r)
    self.assertTrue((len(frame.shape) in (2,3)))

if __name__ == "__main__":
  unittest.main()
