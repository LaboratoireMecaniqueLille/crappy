#coding: utf-8

from videoextenso import Video_extenso
import cv2

ve = Video_extenso(white_spots=False,spots=2)

cap = cv2.VideoCapture(cv2.CAP_XIAPI)

cap.set(cv2.CAP_PROP_XI_AEAG,0)
cap.set(cv2.CAP_PROP_XI_EXPOSURE,10000)
cv2.namedWindow("img",cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

while True:
  r,f = cap.read()
  ve.detect_spots(f[1024-200:1224,1024-300:1324],1024-200,1024-300)
  l = ve.spot_list
  for spot in l:
    miny,minx,maxy,maxx = spot['bbox']
    f[miny:maxy,minx] = 255
    f[miny:maxy+1,maxx] = 255
    f[miny,minx:maxx] = 255
    f[maxy,minx:maxx+1] = 255
    f[spot['y'],spot['x']] = 255
  cv2.imshow("img",f)
  cv2.waitKey(5)
  if raw_input("Ok?").lower() == "y":
    break

ve.save_length()
#ve.start_tracking()

while True:
  f = cap.read()[1]
  print("def:",ve.get_def(f))
  l = ve.spot_list
  for spot in l:
    miny,minx,maxy,maxx = spot['bbox']
    try:
      f[miny:maxy,minx] = 255
      f[miny:maxy,maxx] = 255
      f[miny,minx:maxx] = 255
      f[maxy,minx:maxx] = 255
    except:
      pass
  cv2.imshow("img",f)
  cv2.waitKey(5)
  #if raw_input("continue?").lower() == "n":
  #  break
ve.stop_tracking()


