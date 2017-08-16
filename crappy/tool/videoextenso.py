#coding: utf-8


from multiprocessing import Process, Pipe
import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.morphology import label
from skimage.measure import regionprops

class LostSpotError(Exception):
  pass

def overlapping(box1,box2):
  """Returns True if box1 and box2 are overlapping or included in each other"""
  for i in box1[::2]:
    if box2[0] < i < box2[2]:
      if not (box1[3] <= box2[1] or box2[3] <= box1[1]):
        return True
  for i in box1[1::2]:
    if box2[1] < i < box2[3]:
      if not (box1[2] <= box2[0] or box2[2] <= box1[0]):
        return True
  #Inclusion
  for b1,b2 in ((box1,box2),(box2,box1)):
    if (b1[0] <= b2[0] <= b2[2] <= b1[2]) and (b1[1] <= b2[1] <= b2[3] <= b1[3]):
      return True
  return False

class Video_extenso(object):
  """
The basic VideoExtenso class:
  Can take 1, 2 or 4 spots
  It will detect the spots, save the initial position, and return the
  measured deformation in the most simple way:
  It will always return a list of the spots coordinate (in pixel)
    It will return Exx, Eyy, projections of the length of the bounding
    box of the spot on each axis, divided by its original length
  """
  def __init__(self,**kwargs):
    for arg,default in [("white_spots",False),
                        # Set to True if spots are lighter than the
                        # surroundings, else set to False
                        ("update_thresh",False),
                        # Should the threshold be updated in each round ?
                        # lower chance to loose the spots, but more noise
                        # in the measurements
                        ("num_spots","auto"),
                        # Can be either 'auto',2,3 or 4
                        # It will help for spot detection
                        # it allows to force detection of a given number
                        # of spots (auto works fine most of the time)
                        ("safe_mode",False),
                        # If set to False, it will try hard to catch the
                        # spots when losing them.
                        # Could result in incoherent values without crash.
                        # Set it to True when security is a concern.
                        ("border",5)]:
                        # The number of pixel that will be added to the limits
                        # of the boundingbox
      setattr(self,arg,kwargs.pop(arg,default))
    assert not kwargs,"Invalid kwarg in ve:"+str(kwargs)
    assert self.num_spots in ['auto',2,3,4],"Invalid number of spots!"
    self.spot_list = []
    self.fallback_mode = False
    self.consecutive_overlaps = 0
    # This number of pixel will be added to the window sending the
    # spot image to the process

  def detect_spots(self,img,oy,ox):
    """Detect the spots in img, subframe of the full image
    ox and oy represent the offset of the subframe in the full image"""
    # Finding out how many spots we should detect
    # If L0 is already saved, we have already counted the spots, else
    # see the num_spot parameter
    #img = rank.median(img,np.ones((15,15),dtype=img.dtype))
    img = cv2.medianBlur(img,5)
    self.thresh = threshold_otsu(img)
    if self.white_spots:
      bw = img > self.thresh
    else:
      bw = img <= self.thresh
    #bw = dilation(bw,np.ones((3,3),dtype=img.dtype))
    #bw = erosion(bw,np.ones((3,3),dtype=img.dtype))
    bw = label(bw)
    # Is it really useful?
    #bw[0,:] = bw[-1,:] = bw[:,0] = bw[:,-1] = -1
    l = regionprops(bw)
    # Remove the regions that are clearly not spots
    l = [r for r in l if r.solidity > .8 and r.eccentricity < .95]
    # Remove the too small regions (150 is reeaally tiny)
    l = [r for r in l if r.area > 150]
    l = sorted(l,key=lambda r:r.area,reverse=True)
    i = 0
    while i < len(l)-1:
      r1 = l[i]
      for j in range(i+1,len(l)):
        r2 = l[j]
        if overlapping(r1['bbox'],r2['bbox']):
          print("Overlap")
          if r1.area > r2.area:
            del l[j]
          else:
            del l[i]
          i-=1
          break
      i+=1
    if self.num_spots == 'auto':
      # Remove the smallest region until we have a valid number
      # and all of them are larger than 150 pix
      while len(l) not in [0,2,3,4]:
        del l[-1]
      if len(l) == 0:
        print("Not spots found!")
        return
    else:
      if len(l) < self.num_spots:
        print("Found only",len(l),"spots when expecting",self.num_spots)
        return
      l = l[:self.num_spots] # Keep the largest ones
    print("Detected",len(l),"spots")
    self.spot_list = []
    for r in l:
      d = {}
      y,x = r.centroid
      d['y'] = oy + y
      d['x'] = ox + x
      #l1 = r.major_axis_length
      #l2 = r.minor_axis_length
      #s,c = np.sin(r.orientation)**2,np.cos(r.orientation)**2
      #lx = (l1*c+l2*s)/2
      #ly = (l1*s+l2*c)/2
      #d['bbox'] = d['y']-ly,d['x']-lx,d['y']+ly,d['x']+lx
      #d['bbox'] = d['min_col'],d['min_row'],d['max_col'],d['max_row']
      #d['bbox'] = tuple([int(i+.5) for i in d['bbox']])
      d['bbox'] = tuple([r['bbox'][i]+(oy,ox)[i%2] for i in range(4)])
      self.spot_list.append(d)
    print(self.spot_list)

  def save_length(self):
    if not hasattr(self,"spot_list"):
      print("You must select the spots first!")
      return
    if not hasattr(self,"tracker"):
      self.start_tracking()
    y = [s['y'] for s in self.spot_list]
    x = [s['x'] for s in self.spot_list]
    self.l0y = max(y)-min(y)
    self.l0x = max(x)-min(x)
    self.num_spots = len(self.spot_list)

  def enlarged_window(self,window,shape):
    """Returns the slices to get the window around the spot"""
    s1 = slice(max(0,window[0]-self.border),min(shape[0],window[2]+self.border))
    s2 = slice(max(0,window[1]-self.border),min(shape[1],window[3]+self.border))
    return s1,s2

  def start_tracking(self):
    """Will spawn a process per spot, which goal is to track the spot and
    send the new coordinate after each update"""
    self.tracker = []
    self.pipe = []
    for s in self.spot_list:
      i,o = Pipe()
      self.pipe.append(i)
      self.tracker.append(Tracker(o,white_spots=self.white_spots,
                      thresh='auto' if self.update_thresh else self.thresh,
                      safe_mode=self.safe_mode))
      self.tracker[-1].start()

  def get_def(self,img):
    """The "heart" of the videoextenso
    Will keep track of the spots and return the computed deformation"""
    if not hasattr(self,"l0x"):
      print("L0 not saved, saving it now.")
      self.save_length()
    for p,s in zip(self.pipe,self.spot_list):
      win = self.enlarged_window(s['bbox'],img.shape)
      #print("DEBUG: win is",s['bbox'],"sending",win)
      p.send(((win[0].start,win[1].start),img[win]))
    ol = False
    for p,s in zip(self.pipe,self.spot_list):
      r = p.recv()
      if isinstance(r,str):
        self.stop_tracking()
        raise LostSpotError("Tracker returned"+r)
      l = list(self.spot_list)
      l.remove(s)
      # Please excuse me for the following line,
      # understand: "if this box overlaps any existing box"
      if any([overlapping(a_b[0],a_b[1]['bbox']) for a_b in zip([r['bbox']]*len(l),l)]):
        if self.safe_mode:
          print("Overlapping!")
          self.stop_tracking()
          raise LostSpotError("[safe mode] Overlap")
        print("Overlap! Reducing spot window...")
        ol = True
        s['bbox'] = (min(s['bbox'][0]+1,s['y']-2),min(s['bbox'][1]+1,s['x']-2),
                     max(s['bbox'][2]-1,s['y']+2),max(s['bbox'][3]-1,s['x']+2))
        continue
      s.update(r)
      #print("DEBUG updating spot to",s)
    if ol:
      self.consecutive_overlaps += 1
      if self.consecutive_overlaps >= 10:
        print("Too many overlaps, I give up!")
        raise LostSpotError("Multiple overlaps")
    else:
      self.consecutive_overlaps = 0
    y = [s['y'] for s in self.spot_list]
    x = [s['x'] for s in self.spot_list]
    eyy = (max(y)-min(y))/self.l0y - 1
    exx = (max(x)-min(x))/self.l0x - 1
    return [100*eyy,100*exx]

  def stop_tracking(self):
    for p in self.pipe:
      p.send(("","stop"))


class Tracker(Process):
  """Process tracking a spot for videoextensometry."""
  def __init__(self,pipe,white_spots=False,thresh='auto',safe_mode=True):
    Process.__init__(self)
    self.pipe = pipe
    self.white_spots = white_spots
    self.safe_mode = safe_mode
    self.fallback_mode = False
    if thresh == 'auto':
      self.auto_thresh = True
    else:
      self.auto_thresh = False
      self.thresh = thresh

  def run(self):
    #print("DEBUG: process starting, thresh=",self.thresh)
    while True:
      offset,img = self.pipe.recv()
      if type(img) != np.ndarray:
        break
      oy,ox = offset
      r = self.evaluate(img)
      if not isinstance(r,dict):
        r = self.fallback(img)
        if not isinstance(r,dict):
          continue
      else:
        self.fallback_mode = False
      r['y'] += oy
      r['x'] += ox
      miny,minx,maxy,maxx = r['bbox']
      #print("DEBUG: bbox=",r['bbox'])
      r['bbox'] = miny+oy,minx+ox,maxy+oy,maxx+ox
      #print("DEBUG: new bbox=",r['bbox'])
      self.pipe.send(r)
    #print("DEBUG: Process terminating")

  def evaluate(self,img):
    img = cv2.medianBlur(img,5)
    if self.auto_thresh:
      self.thresh = threshold_otsu(img)
    if self.white_spots:
      bw = (img > self.thresh).astype(np.uint8)
    else:
      bw = (img <= self.thresh).astype(np.uint8)
    #cv2.imshow(self.name,bw*255)
    #cv2.waitKey(5)
    if not .1* img.size < np.count_nonzero(bw) < .8*img.size:
      print("reevaluating threshold!!")
      print("Ratio:",np.count_nonzero(bw)/img.size)
      print("old:",self.thresh)
      self.thresh = threshold_otsu(img)
      print("new:",self.thresh)
    m = cv2.moments(bw)
    r = {}
    try:
      r['x'] = m['m10']/m['m00']
      r['y'] = m['m01']/m['m00']
    except ZeroDivisionError:
      return -1
    x,y,w,h = cv2.boundingRect(bw)
    #if (h,w) == img.shape:
    #  return -1
    r['bbox'] = y,x,y+h,x+w
    return r

  def fallback(self,img):
    """Called when the spots are lost"""
    if self.safe_mode or self.fallback_mode:
      if self.fallback_mode:
        self.pipe.send("Fallback failed")
      else:
        self.pipe.send("[safe mode] Could not compute barycenter")
      self.fallback_mode = False
      return -1
    self.fallback_mode = True
    print("Loosing spot! Trying to reevaluate threshold...")
    self.thresh = threshold_otsu(img)
    return self.evaluate(img)
