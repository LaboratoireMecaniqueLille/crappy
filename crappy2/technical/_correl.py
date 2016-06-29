#coding:utf-8
from __future__ import division,print_function

import numpy as np
import warnings
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda.reduction import ReductionKernel
from math import ceil
from scipy import interpolate
import cv2

from crappy2 import __path__ as crappyPath
kernelFile = crappyPath[0]+"/data/kernels.cu"

context = None

#########################################################################
#=======================================================================#
#=                                                                     =#
#=                        Class CorrelStage:                           =#
#=                                                                     =#
#=======================================================================#
#########################################################################

class CorrelStage:
  """
  Run a correlation routine on an image, at a given resolution. Multiple instances of this class are used for the pyramidal correlation in Correl().
  Can but is not meant to be used as is.
  """
  num = 0 # To count the instances so they get a unique number (self.num), for easier debugging
  def __init__(self,img_size,**kwargs):
    self.num = CorrelStage.num
    CorrelStage.num+=1
    self.verbose = kwargs.get("verbose",0)
    self.debug(2,"Initializing with resolution",img_size)
    self.w,self.h = img_size
    self.__ready = False
    self.nbIter = kwargs.get("iterations",5)
    self.showDiff=kwargs.get("showDiff",False)
    if self.showDiff:
      import cv2
      cv2.namedWindow("Correlation",cv2.WINDOW_NORMAL)

    self.rX,self.rY = -1,-1
    self.loop = 0

    ### Allocating stuff ###

    ### Grid and block for kernels called with the size of the image ##
    self.grid = (int(ceil(self.w/32)),int(ceil(self.h/32)))
    self.block = (int(ceil(self.w/self.grid[0])),int(ceil(self.h/self.grid[1])),1)
    self.debug(3,"Default grid:",self.grid,"block",self.block)
    
    ### We need to know the number of fields to allocate the right number of G tables ###
    self.Nfields = kwargs.get("Nfields")
    if self.Nfields is None:
      self.Nfields = len(kwargs.get("fields")[0])

    ### Allocating everything we need ###
    self.devG = []
    self.devFieldsX = []
    self.devFieldsY = []
    for i in range(self.Nfields):
      self.devG.append(gpuarray.empty(img_size,np.float32)) # To store the G arrays (to compute the research direction)
      self.devFieldsX.append(gpuarray.empty((self.w,self.h),np.float32)) # To store the fields
      self.devFieldsY.append(gpuarray.empty((self.w,self.h),np.float32))
    self.H = np.zeros((self.Nfields, self.Nfields),np.float32)
    self.devHi = gpuarray.empty((self.Nfields,self.Nfields),np.float32)
    self.devOut = gpuarray.empty((self.w,self.h),np.float32) # To store the diff table
    self.devX = gpuarray.empty((self.Nfields),np.float32) # To store the current value of the parameters
    self.devVec = gpuarray.empty((self.Nfields),np.float32) # to store the research direction
    self.devOrig = gpuarray.empty(img_size,np.float32) #gpuarray for original image
    self.devDiff = gpuarray.empty(img_size,np.float32) #gpuarray for the difference
    self.devGradX = gpuarray.empty(img_size,np.float32) #for the gradient along X
    self.devGradY = gpuarray.empty(img_size,np.float32) #...and along Y

    ### Reading kernels and compiling module ###
    with open(kernelFile,"r") as f:
      self.debug(3,"Sourcing module")
      self.mod = SourceModule(f.read()%(self.w,self.h,self.Nfields))

    ### Assigning functions to the kernels ###
    self.__resampleOrigKrnl = self.mod.get_function('resampleO')
    self.__resampleKrnl = self.mod.get_function('resample')
    self.__gradientKrnl = self.mod.get_function('gradient')
    self.__makeGKrnl = self.mod.get_function('makeG')
    self.__makeDiff = self.mod.get_function('makeDiff')
    self.__dotKrnl = self.mod.get_function('myDot')
    self.__addKrnl = self.mod.get_function('kadd')
    self.__mulRedKrnl = ReductionKernel(np.float32, neutral="0", reduce_expr="a+b", map_expr="x[i]*y[i]", arguments = "float *x, float *y")
    self.__leastSquare = ReductionKernel(np.float32, neutral="0", reduce_expr="a+b", map_expr="x[i]*x[i]", arguments = "float *x") # Could use mulRedKrnl(x,x) but probably faster ?

    ### Getting texture references ###
    self.tex = self.mod.get_texref('tex')
    self.tex_d = self.mod.get_texref('tex_d')

    ### Preparing kernels for less overhead when called ###
    self.__resampleOrigKrnl.prepare("Pii",texrefs=[self.tex])
    self.__resampleKrnl.prepare("Pii",texrefs=[self.tex_d])
    self.__gradientKrnl.prepare("PP",texrefs=[self.tex])
    self.__makeDiff.prepare("PPPP",texrefs=[self.tex,self.tex_d])
    self.__addKrnl.prepare("PfP")

    ### Setting proper flags ###
    for t in [self.tex,self.tex_d]:
      t.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
      t.set_filter_mode(cuda.filter_mode.LINEAR)
      t.set_address_mode(0,cuda.address_mode.BORDER)
      t.set_address_mode(1,cuda.address_mode.BORDER)

    ### Reading original image if provided ###
    if kwargs.get("img") is not None:
      self.setOrig(kwargs.get("img"))
    ### Reading fields if provided ###
    if kwargs.get("fields") is not None:
      self.setFields(kwargs.get("fields"))


  def debug(self,n,*s):
    """
    To print debug messages
    First argument is the level of the message. The others arguments will be displayed only if the self.debug var is superior or equal
    """
    if n <= self.verbose:
      s2 = ()
      for i in range(len(s)):
        s2 += (str(s[i]).replace("\n","\n"+(10+n)*" "),)
      print("  "*(n-1)+"[Stage "+str(self.num)+"]",*s2)
    

  def setOrig(self,img):
    """
    To set the original image from a given CPU or GPU array. If it is a GPU array, it will NOT be copied.
    Note that the most efficient method is to write directly over self.devOrig and then run self.updateOrig()
    """
    assert img.shape == (self.w,self.h),"Got a {} image in a {} correlation routine!".format(img.shape,(self.w,self.h))
    if isinstance(img,np.ndarray):
      self.debug(3,"Setting original image from ndarray")
      self.devOrig.set(img)
    elif isinstance(img,gpuarray.GPUArray):
      self.debug(3,"Setting original image from GPUArray")
      self.devOrig = img
    else:
      print("[Error] Unknown type of data given to CorrelStage.setOrig()")
      raise ValueError
    self.updateOrig()

  def updateOrig(self):
    """
    Needs to be called after self.img_d has been written directly (without using setOrig)
    """
    self.debug(3,"Updating original image")
    self.array = cuda.gpuarray_to_array(self.devOrig,"C")
    self.tex.set_array(self.array)
    self.__computeGradients()
    self.__ready = False
      
  def __computeGradients(self):
    self.__gradientKrnl.prepared_call(self.grid,self.block,self.devGradX.gpudata,self.devGradY.gpudata)


  def prepare(self):
    if not self.__ready:
      if not hasattr(self,'array'):
        self.debug(1,"Tried to prepare but original texture is not set !")
      elif not hasattr(self,'fields'):
        self.debug(1,"Tried to prepare but fields are not set !")
      else:
        self.__makeG()
        self.__makeH()
        self.__ready = True
        self.debug(3,"Ready!")
    else:
      self.debug(1,"Tried to prepare when unnecessary, doing nothing...")
        

  def __makeG(self):
    for i in range(self.Nfields):
      self.__makeGKrnl(self.devG[i].gpudata, self.devGradX.gpudata, self.devGradY.gpudata, self.devFieldsX[i],self.devFieldsY[i],block=self.block,grid=self.grid) # Use prepared call ?
    
  def __makeH(self):
    for i in range(self.Nfields):
      for j in range(i+1):
        self.H[i,j] = self.__mulRedKrnl(self.devG[i],self.devG[j]).get()#/((self.w**2+self.h**2)**0.5)
        if i != j:
          self.H[j,i] = self.H[i,j]
    self.devHi.set(np.linalg.inv(self.H))#*1e-3)
    if self.verbose >= 3: # Looks stupid but prevents a useless devHi copy if nothing is printed
      self.debug(3,"Inverted Hessian:\n",self.devHi.get())

  def resampleOrig(self,newX,newY,devOut):
    """
    Reads orig.texture and writes to the devOut array the interpolated newX*newY image
    """
    grid = (int(ceil(newX/32)),int(ceil(newY/32)))
    block = (int(ceil(newX/grid[0])),int(ceil(newY/grid[1])),1)
    self.debug(3,"Resampling Orig texture, grid:",grid,"block:",block)
    self.__resampleOrigKrnl.prepared_call(self.grid,self.block,devOut.gpudata,np.int32(newX),np.int32(newY))

  def resampleD(self, newX,newY):
    """
    Resamples tex_d and returns it in a gpuarray 
    """
    if (self.rX,self.rY) != (np.int32(newX),np.int32(newY)):
      self.rGrid = (int(ceil(newX/32)),int(ceil(newY/32)))
      self.rBlock = (int(ceil(newX/self.rGrid[0])),int(ceil(newY/self.rGrid[1])),1)
      self.rX,self.rY = np.int32(newX),np.int32(newY)
      self.devROut = gpuarray.empty((newX,newY),np.float32)
    self.debug(3,"Resampling img_d texture, grid:",self.rGrid,"block:",self.rBlock)
    self.__resampleKrnl.prepared_call(self.rGrid,self.rBlock,self.devROut.gpudata,self.rX,self.rY)
    return self.devROut

  def setFields(self,fieldsX,fieldsY):
    """
  Takes a tuple/list of 2 (gpu)arrays[Nfields,x,y] (one for displacement along x and one along y) 
  """
    self.debug(2,"Setting fields")
    if isinstance(fieldsX,np.ndarray):
      self.devFieldsX.set(fieldsX)
      self.devFieldsY.set(fieldsY)
    elif isinstance(fieldsX,gpuarray.GPUArray):
      self.devFieldsX = fieldsX
      self.devFieldsY = fieldsY
    self.fields = True

  def setImage(self,img_d):
    """
    Set the image to compare with the original
    """
    assert img_d.shape == (self.w,self.h),"Got a {} image in a {} correlation routine!".format(img_d.shape,(self.w,self.h))
    if isinstance(img_d,np.ndarray):
      self.debug(3,"Creating texture from numpy array")
      self.array_d = cuda.matrix_to_array(img_d,"C")
    elif isinstance(img_d,gpuarray.GPUArray):
      self.debug(3,"Creating texture from gpuarray")
      self.array_d = cuda.gpuarray_to_array(img_d,"C")
    else:
      print("[Error] Unknown type of data given to CorrelStage.setImage()")
      raise ValueError
    self.tex_d.set_array(self.array_d)
    self.devX.set(np.zeros(self.Nfields,dtype=np.float32))

  def setDisp(self,X):
    assert X.shape == (self.Nfields,),"Incorrect initialization of the parameters"
    if isinstance(X,gpuarray.GPUArray):
      self.devX=X
    elif isinstance(X,np.ndarray):
      self.devX.set(X)
    else:
      print("[Error] Unknown type of data given to CorrelStage.setDisp()")
      raise ValueError

  def writeDiffFile(self):
      self.__makeDiff.prepared_call(self.grid,self.block,self.devOut.gpudata,self.devX.gpudata,self.devFieldsX.gpudata,self.devFieldsY.gpudata)
      diff = (self.devOut.get()+128).astype(np.uint8)
      cv2.imwrite("/home/vic/diff/diff{}-{}.png".format(self.num,self.loop),diff)

  def getDisp(self,img_d=None):
    """
    The method that actually computes the weight of the fields.
    """
    self.debug(3,"Calling main routine")
    self.loop += 1
    self.mul = 3 # This parameter is ESSENTIAL. It defines how "fast" we move towards the solution. High value => Fast convergence but risk to go past the solution and diverge (the program does not try to handle this: if the residual rises, iterations stop immediatly). Low value => Probably more precise but slower and may require more iterations. After multiple tests, 3 was found to be a pretty acceptable value. Don't hesitate to adapt it to your case: use verbose=3 and see if the convergence is too slow or too fast.
    if not self.__ready:
      self.debug(2,"Wasn't ready ! Preparing...")
      self.prepare()

    if img_d is not None:
      self.setImage(img_d)

    #self.devX.set(np.array([0]*self.Nfields,dtype=np.float32)) #Already done in setImage()
    assert hasattr(self,'array_d'), "Did not set the image, use setImage() before calling getDisp or give the image as parameter !"
    self.debug(3,"Computing first diff table")
    self.__makeDiff.prepared_call(self.grid,self.block,self.devOut.gpudata,self.devX.gpudata,self.devFieldsX.gpudata,self.devFieldsY.gpudata) # 
    self.res = self.__leastSquare(self.devOut).get()
    self.debug(3,"res:",self.res/1e6)

    ### Iterating ###
    # Note: I know this section is dense and wrappers for kernel calls would have made things clearer, but function calls in python cause a non-negligeable overhead and this is the critical part. The comments are here to guide you !
    for i in range(self.nbIter):
      self.debug(3,"Iteration",i)
      for i in range(self.Nfields): # Computing the gradient of each parameter
        self.devVec[i] = self.__mulRedKrnl(self.devG[i],self.devOut)

      self.__dotKrnl(self.devHi,self.devVec,grid=(1,1),block=(self.Nfields,1,1)) # Newton method: we multiply the gradient vector by the pre-inverted Hessian, devVec now contains the actual research direction.

      #This line simply adds k times the research direction to devX with a really simple kernel (does self.devX += k*self.devVec)
      self.__addKrnl.prepared_call((1,1),(self.Nfields,1,1),self.devX.gpudata,self.mul,self.devVec.gpudata) 

      if self.verbose >= 3: # Do not get rid of this condition: it will not change the output but the vectors will be uselessly copied from the device
        self.debug(3,"Direction:",self.devVec.get())
        self.debug(3,"New X:",self.devX.get())
      
      # To see the new residual
      self.__makeDiff.prepared_call(self.grid,self.block,self.devOut.gpudata,self.devX.gpudata,self.devFieldsX.gpudata,self.devFieldsY.gpudata)

      oldres = self.res
      self.res = self.__leastSquare(self.devOut).get()
      if self.res >= oldres: # If we moved away, revert changes and stop iterating
        self.debug(3,"Diverting from the solution ! Undoing...")
        self.__addKrnl.prepared_call((1,1),(self.Nfields,1,1),self.devX.gpudata,-self.mul,self.devVec.gpudata) 
        self.res = oldres
        break;

      self.debug(3,"res:",self.res/1e6)
    #self.writeDiffFile()
    if self.showDiff:
      cv2.imshow("Correlation",(self.devOut.get()+128).astype(np.uint8))
      cv2.waitKey(1)
    return self.devX.get()


#########################################################################
#=======================================================================#
#=                                                                     =#
#=                          Class Correl:                              =#
#=                                                                     =#
#=======================================================================#
#########################################################################


class TechCorrel:
  """
  This class is the core of the Correl block.
  It is meant to identify efficiently the displacement between two images.

  REQUIREMENTS:
    - The computer must have a Nvidia video card with compute capability 3.0 or higher
    - CUDA 5.0 or higher (only tested with CUDA 7.5)
    - pycuda 2014.1 or higher (only tested with pycuda 2016.1.1)

  PRESENTATION:
    At initialization, TechCorrel needs only one unammed argument: the working resolution (as a tuple of ints), which is the resolution of the images it will be given. All the images must have exactly these dimensions. 
    At initialization or after, this class takes a reference image. The deformations on this image are supposed to be all equal to 0.
    It also needs a number of deformation fields (technically limited to 508 fields, probably less depending on the resolution and the amount of memory on the graphics card).
    Finally, you need to provide the deformed image you want to process
    It will then identify parameters of the sum of fields that lowers the square sum of differences between the original image and the second one displaced with the resulting field.
    This class will resample the images and perform identification on a lower resolution, use the result to initialize the next stage, and again util it reaches the last stage. It will then return the computed parameters. The number of levels can be set with levels=x (see USAGE)
    The latest parameters returned (if any) are used to initialize computation when called again, to help identify large displacement. It is particularly adapted to slow deformations.
    To lower the residual, this program computes the gradient of each parameter and uses Newton method to converge as fast as possible. The number of iterations for the resolution can also be set (see USAGE).


  USAGE:
    The constructor can take a variety of arguments:
    verbose=x with x in [0,1,2,3] is used to choose the amount of information output to the console.
    0: Nothing except for errors
    1: Only important infos and warnings
    2: Major info and a few values periodacally (at a bearable rate)
    3: Tons of info including details of each iteration /!\ Really slows the program down. To be used only for debug.

    ## Fields ##
      Use fields=['x','y',(MyFieldX,MyFieldY),...] To set the fields. Can also be done later with .setFields(...)
      In case where the fields are set later, you need to add Nfields=x to specifiy at init the number of expected fields because it allocates all the necessary memory on the device.
      The fields should be given as a list of tuples (or lists if you wish) of 2 numpy.ndarrays OR gpuarray.GPUArray of the size of the image, each array corresponds to the displacement in pixel along respectively X and Y
    You can also use a string instead of the tuple for common fields:
      - 'x': Movement along X
      - 'y': Movement along Y
      - 'r': Rotation (in the trigonometric direction)
      - 'exx': Stretch along X
      - 'eyy': Stretch along Y
      - 'exy': Shear
      - 'z': Zoom (dilatation)
      Note that you should not try to identify r,exy AND z at the same time (one of them is redundant)
      All of these default fields are normalized to have a max displacement of 1 pixel and are centered in the middle of the image. They are generated to have the correct size for your image.

      For example, to have the stretch in %, simply divide the value by HALF the dimension in pixel along this axis (because it goes from -1 to 1)

    ## Original image ##
      It must be given as a 2D numpy.ndarray. This block works with dtype=np.float32. If the dtype of the given image is different, it will print a warning and the image will be converted.
      It can be given at init with img=MyImage or later with setOrig(MyImage). Note that you can reset it whenever you want, even multiple times but it will reset the def parameters to 0.

      Once fields and original image are set, there is a short preparation time before correlation can append. You can do this preparation yourself by using .prepare().
      If .prepare() is not called, it will be done automatically when necessary, inducing a slight overhead at the first call of .getDisp() after setting/updating the fields or original image

    ## Compared image ##
      It can be given directly when querying the displacement parameters with .getDisp(MyImage) or before calling .getDisp() with setImage(MyImage)
      You can provide it as a np.ndarray just like orig, or as a pycuda.gpuarray.GPUArray.

    ## Editing the behavior ##
      list of kwargs:
        - levels=x (int, x>= 1, default: 5)
          Number of levels of the pyramid (will create x stages)
          More levels can help converging with large and quick deformations/movements but may fail on images without low spatial frequency
          Less levels: program will run faster

        - Resampling_factor=x (float, 1<x, default: 2)
          The resolution will be divided by x between each stage of the pyramid
          Low: Can allow coherence between stages but is more expensive
          High: Reach small resolutions in less levels -> faster but be careful not to loose consistency between stages

        - iterations=x (int, x>=1, default: 4)
          The MAXIMUM number of iteration to be run before returning the values. Note that if the residual increases before reaching x iterations, the block will return anyways.

        - img=x (numpy.ndarray, x.shape=img_size, default: None)
          If you want to set the original image at init.


##################### TO COMPLETE



  TODO:
    This section lists all the considered improvements for this program. These features may NOT all be implemented in the future. They are sorted by priority.
    - Add square deformations to the default fields
    - Add a parameter to return values in %
    - Add a drop parameter to drop images in the link if correlation is not fast enough
    - Add a res parameter to add the residual to the output values
    - Add a filter to smooth/ignore incorrect values
    - Allow faster execution by executing the reduction only on a part of the images (random or chosen)
    - Allow a reset of the reference picture for simple deformations (to enhance robustness in case of large deformations or lightning changes)
    - Restart iterating from 0 once in a while to see if the residual is lower. Can be useful to recover when diverged critically due to an incorrect image (Shadow, obstruction, flash, camera failure, ...)
  """
  def __init__(self,img_size,**kwargs):
    cuda.init()
    from pycuda.tools import make_default_context
    global context
    context = make_default_context()
    self.verbose = kwargs.get("verbose",0)
    self.debug(3,"You set the verbose level to the maximum. \
It may help finding bugs or tracking errors but it may also impact the program performance \
as it will print A LOT of output and add GPU->CPU copies only to print information. \
If it is not desired, consider lowering the verbosity: \
1 or 2 is a reasonable choice, 0 won't show anything except for errors.")
    self.levels = kwargs.get("levels",5)
    self.loop = 0
    self.resamplingFactor = kwargs.get("resampling_factor",2)
    w,h = img_size
    self.nbIter = kwargs.get("iterations",4)
    self.debug(1,"Initializing... Master resolution:",img_size,"levels:",self.levels,"verbosity:",self.verbose)

    ### Computing dimensions of the different levels ###
    self.w,self.h = [],[]
    for i in range(self.levels):
      self.w.append(int(round(w/(self.resamplingFactor**i))))
      self.h.append(int(round(h/(self.resamplingFactor**i))))

    if kwargs.get("Nfields") is not None:
      self.Nfields = kwargs.get("Nfields")
    else:
      try:
        self.Nfields = len(kwargs["fields"])
      except KeyError:
        print("[Error] Correl needs to know the number of fields at init. Add Nfields=x or directly set fields with fields=array")
        raise ValueError

    ### Creating a new instance of CorrelStage for each stage ###
    self.correl=[]
    for i in range(self.levels):
      self.correl.append(CorrelStage((self.w[i],self.h[i]),verbose=self.verbose,Nfields=self.Nfields,iterations=self.nbIter,showDiff=(i==0 and kwargs.get("showDiff",False))))

    ### Set original image if provided ###
    if kwargs.get("img") is not None:
      self.setOrig(kwargs.get("img"))

    s = """
    texture<float, cudaTextureType2D, cudaReadModeElementType> texFx{0};
    texture<float, cudaTextureType2D, cudaReadModeElementType> texFy{0};
    __global__ void resample{0}(float* outX, float* outY, int x, int y)
    {{
      int idx = blockIdx.x*blockDim.x+threadIdx.x;
      int idy = blockIdx.y*blockDim.y+threadIdx.y;
      if(idx < x && idy < y)
      {{
        outX[idy*x+idx] = tex2D(texFx{0},(float)idx/x, (float)idy/y);
        outY[idy*x+idx] = tex2D(texFy{0},(float)idx/x, (float)idy/y);
      }}
    }}
    """
    self.src = ""
    for i in range(self.Nfields):
      self.src+=s.format(i) # Adding textures for the quick fields resampling

    self.mod = SourceModule(self.src)

    self.texFx = []
    self.texFy = []
    self.resampleF = []
    for i in range(self.Nfields):
      self.texFx.append(self.mod.get_texref("texFx%d"%i))
      self.texFy.append(self.mod.get_texref("texFy%d"%i))
      self.resampleF.append(self.mod.get_function("resample%d"%i))
      self.resampleF[i].prepare("PPii",texrefs=[self.texFx[i],self.texFy[i]])

    for t in self.texFx+self.texFy:
      t.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
      t.set_filter_mode(cuda.filter_mode.LINEAR)
      t.set_address_mode(0,cuda.address_mode.CLAMP)
      t.set_address_mode(1,cuda.address_mode.CLAMP)

    ### Set fields if provided ###
    if kwargs.get("fields") is not None:
      self.setFields(kwargs.get("fields"))

  def getFields(self,x,y):
    outX = gpuarray.empty((self.Nfields,x,y),np.float32)
    outY = gpuarray.empty((self.Nfields,x,y),np.float32)
    grid = (int(ceil(x/32)),int(ceil(y/32)))
    block = (int(ceil(x/grid[0])),int(ceil(y/grid[1])),1)
    for i in range(self.Nfields):
      self.resampleF[i].prepared_call(grid,block,outX[i,:,:].gpudata,outY[i,:,:].gpudata,np.int32(x),np.int32(y))
    return outX,outY

  def debug(self,n,*s):
    """
    First argument is the level of the message. It wil be displayed only if the self.debug is superior or equal
    """
    if n <= self.verbose:
      print("  "*(n-1)+"[Correl]",*s)
    
  def setOrig(self,img):
    """
    To set the original image (the reference with which the deformed image will be compared)
    """
    self.debug(2,"updating original image")
    assert isinstance(img,np.ndarray),"Image must be a numpy array"
    assert len(img.shape) == 2,"Image must have 2 dimensions (got {})".format(len(img.shape))
    assert img.shape == (self.w[0],self.h[0]),"Wrong size!"
    if img.dtype != np.float32:
      warnings.warn("Correl() takes arrays with dtype np.float32 to allow GPU computing (got {}). Converting to float32.".format(img.dtype),RuntimeWarning)
      img=img.astype(np.float32)

    self.correl[0].setOrig(img)
    for i in range(1,self.levels):
      self.correl[i-1].resampleOrig(self.w[i],self.h[i],self.correl[i].devOrig)
      self.correl[i].updateOrig()

  def setFields(self,fields):
    assert self.Nfields == len(fields),"Cannot change the number of fields on the go!"
    if isinstance(fields[0], str) or isinstance(fields[0][0], np.ndarray):
      toArray = cuda.matrix_to_array
    elif isinstance(fields[0][0], gpuarray.GPUArray): # Choosing the right function to copy
      toArray = cuda.gpuarray_to_array
    else:
      print("[Correl] Error: Incorrect fields argument. See docstring of Correl")
      raise ValueError
    self.fieldsXArray = [] # These list store the arrays for the fields texture (to be interpolated quickly for each stage)
    self.fieldsYArray = []
    for i in range(self.Nfields):
      if isinstance(fields[i],str):
        c = fields[i].lower()
        if c in ['x','mx','tx']:  #Movement along X
          fields[i] = (np.ones((self.w[0],self.h[0]),dtype=np.float32),np.zeros((self.w[0],self.h[0]),dtype=np.float32))
        elif c in ['y','my','ty']:  #..along Y
          fields[i] = (np.zeros((self.w[0],self.h[0]),dtype=np.float32),np.ones((self.w[0],self.h[0]),dtype=np.float32))
        elif c == 'r': # Rotation
          sq = .5**.5
          z = np.meshgrid(np.arange(-sq,sq,2*sq/self.w[0],dtype=np.float32),np.arange(-sq,sq,2*sq/self.h[0],dtype=np.float32))
          fields[i] = (z[1].astype(np.float32),-z[0].astype(np.float32))
        elif c in ['ex','exx']:  # Stretch along X
          fields[i] = (np.concatenate((np.arange(-1,1,2./self.w[0],dtype=np.float32)[np.newaxis,:],)*self.h[0],axis=0),np.zeros((self.w[0],self.h[0]),dtype=np.float32))
        elif c in ['ey','eyy']: # Stretch along Y
          fields[i] = (np.zeros((self.w[0],self.h[0]),dtype=np.float32),np.concatenate((np.arange(-1,1,2./self.w[0],dtype=np.float32)[:,np.newaxis],)*self.h[0],axis=1))
        elif c in ['exy','tau']: # Shear
          sq = .5**.5
          z = np.meshgrid(np.arange(-sq,sq,2*sq/self.w[0],dtype=np.float32),np.arange(-sq,sq,2*sq/self.h[0],dtype=np.float32))
          fields[i] = (z[1].astype(np.float32),z[0].astype(np.float32))
        elif c == 'z' or c in ['mz','tz']: # Shrinking/Zooming
          sq = .5**.5
          z = np.meshgrid(np.arange(-sq,sq,2*sq/self.w[0],dtype=np.float32),np.arange(-sq,sq,2*sq/self.h[0],dtype=np.float32))
          fields[i] = (z[0].astype(np.float32),z[1].astype(np.float32))
        else:
          print("[Correl] Error: Unrecognized field parameter:",fields[i])
          raise ValueError

      self.fieldsXArray.append(toArray(fields[i][0],"C"))
      self.texFx[i].set_array(self.fieldsXArray[i])
      self.fieldsYArray.append(toArray(fields[i][1],"C"))
      self.texFy[i].set_array(self.fieldsYArray[i])
    """
    print("FIELDS:")
    for f in fields:
      print("X\n",f[0][::256,::256],"\n\n\n")
      print("Y\n",f[1][::256,::256],"\n\n\n")
    """
    for i in range(self.levels):
      self.correl[i].setFields(*self.getFields(self.w[i],self.h[i]))

  def prepare(self):
    for c in self.correl:
      c.prepare()
    self.debug(2,"Ready!")

  def saveAllImages(self,name="out"):
    import cv2
    self.debug(1,"Saving all images with the name",name+"X.png")
    for i in range(self.levels):
      out = self.correl[i].devOrig.get().astype(np.uint8)
      cv2.imwrite(name+str(i)+".png",out)

  def setImage(self,img_d):
    self.correl[0].setImage(img_d)
    for i in range(1,self.levels):
      self.correl[i].setImage(self.correl[i-1].resampleD(self.correl[i].w,self.correl[i].h))

  def getDisp(self,img_d=None):
    self.loop += 1
    if img_d is not None:
      self.setImage(img_d)
    try:
      disp = self.last/(self.resamplingFactor**self.levels)
    except:
      disp = np.array([0]*self.Nfields,dtype=np.float32)
    for i in reversed(range(self.levels)):
      disp *= self.resamplingFactor
      self.correl[i].setDisp(disp)
      disp = self.correl[i].getDisp()
      self.last = disp
    if self.loop % 10 == 0:
      self.debug(2,"Loop",self.loop,", values:",self.correl[0].devX.get(),", res:",self.correl[0].res/1e6)
    return disp

  def getRes(self,lvl=0):
    """
    Returns the last residual of the sepcified level (0 by default)
    Usually, the correlation is correct when res < ~1e9-10 but it really depends on the images: you need to find the value that suit your own images, depending on the resolution, contrast, correlation method etc... You can use writeDiffFile to visualize the difference between the two images after correlation.
    """
    return self.correl[lvl].res


  def writeDiffFile(self,level=0):
    """
    To see the difference between the two images with the computed parameters. It writes a single channel picture named "diff.png" where 128 gray is exact equality, lighter pixels show positive difference and darker pixels a negative difference. Useful to see if correlation succeded and to identify the origin of non convergence
    """
    self.correl[level].writeDiffFile()


if __name__ == "__main__":
  """
  This section is for testing with images from the drive
  """

  import cv2
  from time import sleep,time

  addr = "../Images/ref2.png"
  img = cv2.imread(addr,0)
  if img is None:
    raise IOError("Failed to open "+addr)
  img = img.astype(np.float32)
  x,y = img.shape
  addr_d = "../Images/ref2_grm.png"
  img_d = cv2.imread(addr_d,0)
  if img_d is None:
    raise IOError("Failed to open "+addr_d)
  img_d = img_d.astype(np.float32)



  ones = np.ones((x,y),dtype=np.float32)
  zeros = np.zeros((x,y),dtype=np.float32)

  mvX = (ones,zeros)
  mvY = (zeros,ones)

  sq = .5**.5

  Zoom = np.meshgrid(np.arange(-sq,sq,sq/1024,dtype=np.float32),np.arange(-sq,sq,sq/1024,dtype=np.float32))

  Zoom = (Zoom[0].astype(np.float32),Zoom[1].astype(np.float32))
  Rot = (Zoom[1],-Zoom[0])

  c = Correl((2048,2048),img=img,fields=(mvX,mvY,Rot,Zoom),verbose=3,iterations=6,levels=5)
  c.prepare()

  n = 200
  t1 = time()
  print(c.getDisp(img_d))
  for i in range(n):
    c.getDisp(img_d)
  t2 = time()
  print("Elapsed:",1000*(t2-t1),"ms.")
  print(1000/n*(t2-t1),"ms/iteration.")
