# coding:utf-8

import warnings
from math import ceil
import numpy as np
from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")

from .fields import get_field
from .._global import OptionalModule
try:
  import pycuda.driver as cuda
  from pycuda.compiler import SourceModule
  import pycuda.gpuarray as gpuarray
  from pycuda.reduction import ReductionKernel
except ImportError:
  cuda = OptionalModule("pycuda",
      "PyCUDA and CUDA are necessary to use GPUCorrel")
  SourceModule = OptionalModule("pycuda",
                        "PyCUDA and CUDA are necessary to use GPUCorrel")
  gpuarray = OptionalModule("pycuda",
                        "PyCUDA and CUDA are necessary to use GPUCorrel")
  ReductionKernel = OptionalModule("pycuda",
                        "PyCUDA and CUDA are necessary to use GPUCorrel")


context = None


def interp_nearest(ary, ny, nx):
  """Used to interpolate the mask for each stage."""

  if ary.shape == (ny, nx):
    return ary
  y, x = ary.shape
  rx = x / nx
  ry = y / ny
  out = np.empty((ny, nx), dtype=np.float32)
  for j in range(ny):
    for i in range(nx):
      out[j, i] = ary[int(ry * j + .5), int(rx * i + .5)]
  return out


# =======================================================================#
# =                                                                     =#
# =                        Class CorrelStage:                           =#
# =                                                                     =#
# =======================================================================#

class CorrelStage:
  """Run a correlation routine on an image, at a given resolution.

  Note:
    Multiple instances of this class are used for the pyramidal correlation in
    `Correl()`.

    Can but is not meant to be used as is.
  """

  num = 0  # To count the instances so they get a unique number (self.num)

  def __init__(self, img_size, **kwargs):
    self.num = CorrelStage.num
    CorrelStage.num += 1
    self.verbose = kwargs.get("verbose", 0)
    self.debug(2, "Initializing with resolution", img_size)
    self.h, self.w = img_size
    self._ready = False
    self.nbIter = kwargs.get("iterations", 5)
    self.showDiff = kwargs.get("show_diff", False)
    if self.showDiff:
      try:
        import cv2
      except (ModuleNotFoundError, ImportError):
        cv2 = OptionalModule("opencv-python")
      cv2.namedWindow("Residual", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    self.mul = kwargs.get("mul", 3)
    # These two store the values of the last resampled array
    # It is meant to allocate output array only once (see resample_d)
    self.rX, self.rY = -1, -1
    # self.loop will be incremented every time get_disp is called
    # It will be used to measure performance and output some info
    self.loop = 0

    # Allocating stuff #

    # Grid and block for kernels called with the size of the image #
    # All the images and arrays in the kernels will be in order (x,y)
    self.grid = (int(ceil(self.w / 32)),
                 int(ceil(self.h / 32)))
    self.block = (int(ceil(self.w / self.grid[0])),
                  int(ceil(self.h / self.grid[1])), 1)
    self.debug(3, "Default grid:", self.grid, "block", self.block)

    # We need the number of fields to allocate the G tables #
    self.Nfields = kwargs.get("Nfields")
    if self.Nfields is None:
      self.Nfields = len(kwargs.get("fields")[0])

    # Allocating everything we need #
    self.devG = []
    self.devFieldsX = []
    self.devFieldsY = []
    for i in range(self.Nfields):
      # devG stores the G arrays (to compute the research direction)
      self.devG.append(gpuarray.empty(img_size, np.float32))
      # devFieldsX/Y store the fields value along X and Y
      self.devFieldsX.append(gpuarray.empty((self.h, self.w), np.float32))
      self.devFieldsY.append(gpuarray.empty((self.h, self.w), np.float32))
    # devH Stores the Hessian matrix
    self.H = np.zeros((self.Nfields, self.Nfields), np.float32)
    # And devHi stores its invert
    self.devHi = gpuarray.empty((self.Nfields, self.Nfields), np.float32)
    # devOut is written with the difference of the images
    self.devOut = gpuarray.empty((self.h, self.w), np.float32)
    # devX stores the value of the parameters (what is actually computed)
    self.devX = gpuarray.empty(self.Nfields, np.float32)
    # to store the research direction
    self.devVec = gpuarray.empty(self.Nfields, np.float32)
    # To store the original image on the device
    self.devOrig = gpuarray.empty(img_size, np.float32)
    # To store the gradient along X of the original image on the device
    self.devGradX = gpuarray.empty(img_size, np.float32)
    # And along Y
    self.devGradY = gpuarray.empty(img_size, np.float32)

    # Locating the kernel file #
    kernel_file = kwargs.get("kernel_file")
    if kernel_file is None:
      self.debug(2, "Kernel file not specified")
      from crappy import __path__ as crappy_path
      kernel_file = crappy_path[0] + "/data/kernels.cu"
    # Reading kernels and compiling module #
    with open(kernel_file, "r") as f:
      self.debug(3, "Sourcing module")
      self.mod = SourceModule(f.read() % (self.w, self.h, self.Nfields))
    # Assigning functions to the kernels #
    # These kernels are defined in data/kernels.cu
    self._resampleOrigKrnl = self.mod.get_function('resampleO')
    self._resampleKrnl = self.mod.get_function('resample')
    self._gradientKrnl = self.mod.get_function('gradient')
    self._makeGKrnl = self.mod.get_function('makeG')
    self._makeDiff = self.mod.get_function('makeDiff')
    self._dotKrnl = self.mod.get_function('myDot')
    self._addKrnl = self.mod.get_function('kadd')
    # These ones use pyCuda reduction module to generate efficient kernels
    self._mulRedKrnl = ReductionKernel(np.float32, neutral="0",
                                     reduce_expr="a+b", map_expr="x[i]*y[i]",
                                     arguments="float *x, float *y")
    self._leastSquare = ReductionKernel(np.float32, neutral="0",
                                     reduce_expr="a+b", map_expr="x[i]*x[i]",
                                     arguments="float *x")
    # We could have used use mulRedKrnl(x,x), but this is probably faster ?

    # Getting texture references #
    self.tex = self.mod.get_texref('tex')
    self.tex_d = self.mod.get_texref('tex_d')
    self.texMask = self.mod.get_texref('texMask')
    # Setting proper flags #
    # All textures use normalized coordinates except for the mask
    for t in [self.tex, self.tex_d]:
      t.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
    for t in [self.tex, self.tex_d, self.texMask]:
      t.set_filter_mode(cuda.filter_mode.LINEAR)
      t.set_address_mode(0, cuda.address_mode.BORDER)
      t.set_address_mode(1, cuda.address_mode.BORDER)

    # Preparing kernels for less overhead when called #
    self._resampleOrigKrnl.prepare("Pii", texrefs=[self.tex])
    self._resampleKrnl.prepare("Pii", texrefs=[self.tex_d])
    self._gradientKrnl.prepare("PP", texrefs=[self.tex])
    self._makeDiff.prepare("PPPP",
                           texrefs=[self.tex, self.tex_d, self.texMask])
    self._addKrnl.prepare("PfP")
    # Reading original image if provided #
    if kwargs.get("img") is not None:
      self.set_orig(kwargs.get("img"))
    # Reading fields if provided #
    if kwargs.get("fields") is not None:
      self.set_fields(*kwargs.get("fields"))
    # Reading mask if provided #
    if kwargs.get("mask") is not None:
      self.set_mask(kwargs.get("mask"))

  def debug(self, n, *s):
    """To print debug messages.

    Note:
      First argument is the level of the message.

      The others arguments will be displayed only if the `self.debug` var is
      superior or equal.

      Also, flag and indentation reflect respectively the origin and the level
      of the message.
    """

    if n <= self.verbose:
      s2 = ()
      for i in range(len(s)):
        s2 += (str(s[i]).replace("\n", "\n" + (10 + n) * " "),)
      print("  " * (n - 1) + "[Stage " + str(self.num) + "]", *s2)

  def set_orig(self, img):
    """To set the original image from a given CPU or GPU array.

    Warning:
      If it is a GPU array, it will NOT be copied.

    Note:
      The most efficient method is to write directly over `self.devOrig` with
      some kernel and then run :meth:`update_orig`.
    """

    assert img.shape == (self.h, self.w), \
      "Got a {} image in a {} correlation routine!".format(
        img.shape, (self.h, self.w))
    if isinstance(img, np.ndarray):
      self.debug(3, "Setting original image from ndarray")
      self.devOrig.set(img)
    elif isinstance(img, gpuarray.GPUArray):
      self.debug(3, "Setting original image from GPUArray")
      self.devOrig = img
    else:
      self.debug(0, "Error ! Unknown type of data given to set_orig()")
      raise ValueError
    self.update_orig()

  def update_orig(self):
    """Needs to be called after `self.img_d` has been written directly."""

    self.debug(3, "Updating original image")
    self.array = cuda.gpuarray_to_array(self.devOrig, 'C')
    # 'C' order implies tex2D(x,y) will fetch matrix(y,x):
    # this is where x and y are inverted to comply with the kernels order
    self.tex.set_array(self.array)
    self._compute_gradients()
    self._ready = False

  def _compute_gradients(self):
    """Wrapper to call the gradient kernel."""

    self._gradientKrnl.prepared_call(self.grid, self.block,
                                 self.devGradX.gpudata, self.devGradY.gpudata)

  def prepare(self):
    """Computes all necessary tables to perform correlation.

    Note:
      This method must be called everytime the original image or fields are
      set.

      If not done by the user, it will be done automatically when needed.
    """

    if not hasattr(self, 'maskArray'):
      self.debug(2, "No mask set when preparing, using a basic one, "
                    "with a border of 5% the dimension")
      mask = np.zeros((self.h, self.w), np.float32)
      mask[self.h // 20:-self.h // 20, self.w // 20:-self.w // 20] = 1
      self.set_mask(mask)
    if not self._ready:
      if not hasattr(self, 'array'):
        self.debug(1, "Tried to prepare but original texture is not set !")
      elif not hasattr(self, 'fields'):
        self.debug(1, "Tried to prepare but fields are not set !")
      else:
        self._make_g()
        self._make_h()
        self._ready = True
        self.debug(3, "Ready!")
    else:
      self.debug(1, "Tried to prepare when unnecessary, doing nothing...")

  def _make_g(self):
    for i in range(self.Nfields):
      # Change to prepared call ?
      self._makeGKrnl(self.devG[i].gpudata, self.devGradX.gpudata,
                      self.devGradY.gpudata,
                      self.devFieldsX[i], self.devFieldsY[i],
                      block=self.block, grid=self.grid)

  def _make_h(self):
    for i in range(self.Nfields):
      for j in range(i + 1):
        self.H[i, j] = self._mulRedKrnl(self.devG[i], self.devG[j]).get()
        if i != j:
          self.H[j, i] = self.H[i, j]
    self.debug(3, "Hessian:\n", self.H)
    self.devHi.set(np.linalg.inv(self.H))  # *1e-3)
    # Looks stupid but prevents a useless devHi copy if nothing is printed
    if self.verbose >= 3:
      self.debug(3, "Inverted Hessian:\n", self.devHi.get())

  def resample_orig(self, new_y, new_x, dev_out):
    """To resample the original image.

    Note:
      Reads `orig.texture` and writes the interpolated `newX*newY` image to the
      `devOut` array.
    """

    grid = (int(ceil(new_x / 32)), int(ceil(new_y / 32)))
    block = (int(ceil(new_x / grid[0])), int(ceil(new_y / grid[1])), 1)
    self.debug(3, "Resampling Orig texture, grid:", grid, "block:", block)
    self._resampleOrigKrnl.prepared_call(self.grid, self.block,
                                         dev_out.gpudata,
                                         np.int32(new_x), np.int32(new_y))
    self.debug(3, "Resampled original texture to", dev_out.shape)

  def resample_d(self, new_y, new_x):
    """Resamples `tex_d` and returns it in a `gpuarray`."""

    if (self.rX, self.rY) != (np.int32(new_x), np.int32(new_y)):
      self.rGrid = (int(ceil(new_x / 32)), int(ceil(new_y / 32)))
      self.rBlock = (int(ceil(new_x / self.rGrid[0])),
                     int(ceil(new_y / self.rGrid[1])), 1)
      self.rX, self.rY = np.int32(new_x), np.int32(new_y)
      self.devROut = gpuarray.empty((new_y, new_x), np.float32)
    self.debug(3, "Resampling img_d texture to", (new_y, new_x),
               " grid:", self.rGrid, "block:", self.rBlock)
    self._resampleKrnl.prepared_call(self.rGrid, self.rBlock,
                                     self.devROut.gpudata,
                                     self.rX, self.rY)
    return self.devROut

  def set_fields(self, fields_x, fields_y):
    """Method to give the fields to identify with the routine.

    Note:
      This is necessary only once and can be done multiple times, but the
      routine have to be initialized with :meth:`prepare`, causing a slight
      overhead.

      Takes a :obj:`tuple` or :obj:`list` of 2 `(gpu)arrays[Nfields,x,y]` (one
      for displacement along `x` and one along `y`).
    """

    self.debug(2, "Setting fields")
    if isinstance(fields_x, np.ndarray):
      self.devFieldsX.set(fields_x)
      self.devFieldsY.set(fields_y)
    elif isinstance(fields_x, gpuarray.GPUArray):
      self.devFieldsX = fields_x
      self.devFieldsY = fields_y
    self.fields = True

  def set_image(self, img_d):
    """Set the image to compare with the original.

    Note:
      Calling this method is not necessary: you can do `.get_disp(image)`.
      This will automatically call this method first.
    """

    assert img_d.shape == (self.h, self.w), \
      "Got a {} image in a {} correlation routine!".format(
        img_d.shape, (self.h, self.w))
    if isinstance(img_d, np.ndarray):
      self.debug(3, "Creating texture from numpy array")
      self.array_d = cuda.matrix_to_array(img_d, "C")
    elif isinstance(img_d, gpuarray.GPUArray):
      self.debug(3, "Creating texture from gpuarray")
      self.array_d = cuda.gpuarray_to_array(img_d, "C")
    else:
      self.debug(0, "Error ! Unknown type of data given to .set_image()")
      raise ValueError
    self.tex_d.set_array(self.array_d)
    self.devX.set(np.zeros(self.Nfields, dtype=np.float32))

  def set_mask(self, mask):
    self.debug(3, "Setting the mask")
    assert mask.shape == (self.h, self.w), \
      "Got a {} mask in a {} routine.".format(mask.shape, (self.h, self.w))
    if not mask.dtype == np.float32:
      self.debug(2, "Converting the mask to float32")
      mask = mask.astype(np.float32)
    if isinstance(mask, np.ndarray):
      self.maskArray = cuda.matrix_to_array(mask, 'C')
    elif isinstance(mask, gpuarray.GPUArray):
      self.maskArray = cuda.gpuarray_to_array(mask, 'C')
    else:
      self.debug(0, "Error! Mask data type not understood")
      raise ValueError
    self.texMask.set_array(self.maskArray)

  def set_disp(self, x):
    assert x.shape == (self.Nfields,), \
      "Incorrect initialization of the parameters"
    if isinstance(x, gpuarray.GPUArray):
      self.devX = x
    elif isinstance(x, np.ndarray):
      self.devX.set(x)
    else:
      self.debug(0, "Error! Unknown type of data given to "
                    "CorrelStage.set_disp")
      raise ValueError

  def write_diff_file(self):
    self._makeDiff.prepared_call(self.grid, self.block,
                                 self.devOut.gpudata,
                                 self.devX.gpudata,
                                 self.devFieldsX.gpudata,
                                 self.devFieldsY.gpudata)
    diff = (self.devOut.get() + 128).astype(np.uint8)
    cv2.imwrite("/home/vic/diff/diff{}-{}.png"
                .format(self.num, self.loop), diff)

  def get_disp(self, img_d=None):
    """The method that actually computes the weight of the fields."""

    self.debug(3, "Calling main routine")
    self.loop += 1
    # self.mul = 3
    if not self._ready:
      self.debug(2, "Wasn't ready ! Preparing...")
      self.prepare()
    if img_d is not None:
      self.set_image(img_d)
    assert hasattr(self, 'array_d'), \
      "Did not set the image, use set_image() before calling get_disp \
  or give the image as parameter."
    self.debug(3, "Computing first diff table")
    self._makeDiff.prepared_call(self.grid, self.block,
                                 self.devOut.gpudata,
                                 self.devX.gpudata,
                                 self.devFieldsX.gpudata,
                                 self.devFieldsY.gpudata)
    self.res = self._leastSquare(self.devOut).get()
    self.debug(3, "res:", self.res / 1e6)

    # Iterating #
    # Note: I know this section is dense and wrappers for kernel calls could
    # have made things clearer, but function calls in python cause a
    # non-negligible overhead and this is the critical part.
    # The comments are here to guide you !
    for i in range(self.nbIter):
      self.debug(3, "Iteration", i)
      for j in range(self.Nfields):
        # Computing the direction of the gradient of each parameters
        self.devVec[j] = self._mulRedKrnl(self.devG[j], self.devOut)
      # Newton method: we multiply the gradient vector by the pre-inverted
      # Hessian, devVec now contains the actual research direction.
      self._dotKrnl(self.devHi, self.devVec,
                    grid=(1, 1), block=(self.Nfields, 1, 1))
      # This line simply adds k times the research direction to devX
      # with a really simple kernel (does self.devX += k*self.devVec)
      self._addKrnl.prepared_call((1, 1), (self.Nfields, 1, 1),
                                  self.devX.gpudata, self.mul,
                                  self.devVec.gpudata)
      # Do not get rid of this condition: it will not change the output but
      # the parameters will be evaluated, this will copy data from the device
      if self.verbose >= 3:
        self.debug(3, "Direction:", self.devVec.get())
        self.debug(3, "New X:", self.devX.get())

      # To get the new residual
      self._makeDiff.prepared_call(self.grid, self.block,
                                   self.devOut.gpudata,
                                   self.devX.gpudata,
                                   self.devFieldsX.gpudata,
                                   self.devFieldsY.gpudata)
      oldres = self.res
      self.res = self._leastSquare(self.devOut).get()
      # If we moved away, revert changes and stop iterating
      if self.res >= oldres:
        self.debug(3, "Diverting from the solution new res={} >= {}!"
                   .format(self.res / 1e6, oldres / 1e6))
        self._addKrnl.prepared_call((1, 1), (self.Nfields, 1, 1),
                                    self.devX.gpudata,
                                    -self.mul,
                                    self.devVec.gpudata)
        self.res = oldres
        self.debug(3, "Undone: X=", self.devX.get())
        break

      self.debug(3, "res:", self.res / 1e6)
    # self.write_diff_file()
    if self.showDiff:
      cv2.imshow("Residual", (self.devOut.get() + 128).astype(np.uint8))
      cv2.waitKey(1)
    return self.devX.get()


# =======================================================================#
# =                                                                     =#
# =                           Class Correl:                             =#
# =                                                                     =#
# =======================================================================#


class GPUCorrel:
  """Identify the displacement between two images.

  This class is the core of the Correl block. It is meant to be efficient
  enough to run in real-time.

  It relies on :class:`CorrelStage` to perform correlation on different scales.

  Requirements:
    - The computer must have a Nvidia video card with compute capability
      `>= 3.0`
    - `CUDA 5.0` or higher (only tested with `CUDA 7.5`)
    - `pycuda 2014.1` or higher (only tested with pycuda `2016.1.1`)

  Presentation:
    This class takes a :obj:`list` of fields. These fields will be the base of
    deformation in which the displacement will be identified. When given two
    images, it will identify the displacement between the original and the
    second image in this base as closely as possible lowering square-residual
    using provided displacements.

    This class is highly flexible and performs on GPU for faster operation.

  Usage:
    At initialization, Correl needs only one unnamed argument: the working
    resolution (as a :obj:`tuple` of :obj:`int`), which is the resolution of
    the images it will be given. All the images must have exactly these
    dimensions. The dimensions must be given in this order: `(y,x)` (like
    `openCV` images)

    At initialization or after, this class takes a reference image. The
    deformations on this image are supposed to be all equal to `0`.

    It also needs a number of deformation fields (technically limited to `~500`
    fields, probably much less depending on the resolution and the amount of
    memory on the graphics card).

    Finally, you need to provide the deformed image you want to process. It
    will then identify parameters of the sum of fields that lowers the square
    sum of differences between the original image and the second one displaced
    with the resulting field.

    This class will resample the images and perform identification on a lower
    resolution, use the result to initialize the next stage, and again util it
    reaches the last stage. It will then return the computed parameters. The
    number of levels can be set with ``levels=x``.

    The latest parameters returned (if any) are used to initialize computation
    when called again, to help identify large displacement. It is particularly
    adapted to slow deformations.

    To lower the residual, this program computes the gradient of each parameter
    and uses Newton method to converge as fast as possible. The number of
    iterations for the resolution can also be set.

  Args:
    img_size (:obj:`tuple`): tuple of 2 :obj:`int`, `(y,x)`, the working
      resolution

    verbose (:obj:`int`): Use ``verbose=x`` to choose the amount of information
      printed to the console:

      - `0`: Nothing except for errors
      - `1`: Only important info and warnings
      - `2`: Major info and a few values periodically (at a bearable rate)
      - `3`: Tons of info including details of each iteration

      Note that `verbose=3` REALLY slows the program down. To be used only for
      debug.

    fields (:obj:`list`): Use ``fields=[...]`` to set the fields. This can be
      done later with :meth:`set_fields`, however in case when the fields are
      set later, you need to add ``Nfields=x`` to specify at :meth:`__init__`
      the number of expected fields in order to allocate all the necessary
      memory on the device.

      The fields should be given as a :obj:`list` of :obj:`tuple` of 2
      `numpy.ndarrays` or `gpuarray.GPUArray` of the size of the image, each
      array corresponds to the displacement in pixel along respectively `X` and
      `Y`.

      You can also use a :obj:`str` instead of the :obj:`tuple` for the common
      fields:

      - Rigid body and linear deformations:

        - `'x'`: Movement along `X`
        - `'y'`: Movement along `Y`
        - `'r'`: Rotation (in the trigonometric direction)
        - `'exx'`: Stretch along `X`
        - `'eyy'`: Stretch along `Y`
        - `'exy'`: Shear
        - `'z'`: Zoom (dilatation) (`=exx+eyy`)

        Note that you should not try to identify `exx`, `eyy` AND `z` at the
        same time (one of them is redundant).

      - Quadratic deformations:

        These fields are more complicated to interpret but can be useful for
        complicated solicitations such as biaxial stretch. `U` and `V`
        represent the displacement along respectively `x` and `y`.

        - `'uxx'`: `U(x,y) = x²`
        - `'uyy'`: `U(x,y) = y²`
        - `'uxy'`: `U(x,y) = xy`
        - `'vxx'`: `V(x,y) = x²`
        - `'vyy'`: `V(x,y) = y²`
        - `'vxy'`: `V(x,y) = xy`

      All of these default fields are normalized to have a max displacement of
      `1` pixel and are centered in the middle of the image. They are generated
      to have the size of your image.

      You can mix strings and tuples at your convenience to perform your
      identification.

      Example:
        ::

          fields=['x', 'y', (MyFieldX, MyFieldY)]

        where `MyfieldX` and `MyfieldY` are numpy arrays with the same shape as
        the images

        Example of memory usage: On a 2048x2048 image, count roughly
        `180 + 100*Nfields` MB of VRAM

    img: The original image. It must be given as a 2D `numpy.ndarray`. This
      block works with `dtype=np.float32`. If the `dtype` of the given image is
      different, it will print a warning and the image will be converted. It
      can be given at :meth:`__init__` with the kwarg ``img=MyImage`` or later
      with ``set_orig(MyImage)``.

      Note:
        You can reset it whenever you want, even multiple times but it will
        reset the def parameters to `0`.

      Once fields and original image are set, there is a short preparation time
      before correlation can be performed. You can do this preparation yourself
      by using :meth:`prepare`. If not called, it will be done automatically
      when necessary, inducing a slight overhead at the first call of
      :meth:`get_disp` after setting/updating the fields or original image.

    levels (:obj:`int`, optional): Number of levels of the pyramid. More levels
      can help converging with large and quick deformations but may fail on
      images without low spatial frequency. Fewer levels mean that the program
      will run faster.

    resampling_factor (:obj:`float`, optional): The resolution will be divided
      by this parameter between each stage of the pyramid. Low, can allow
      coherence between stages but is more expensive. High, reaches small
      resolutions in less levels and is faster but be careful not to loose
      consistency between stages.

    iterations (:obj:`int`, optional): The MAXIMUM number of iteration to be
      ran before returning the values. Note that if the residual increases
      before reaching `x` iterations, the block will return anyway.

    mask (optional): To set the mask, to weight the zone of interest on the
      images. It is particularly useful to prevent undesired effects on the
      border of the images. If no mask is given, a rectangular mask will be
      used, with border of `5%` the size of the image.

    show_diff (:obj:`bool`, optional): Will open a :mod:`cv2` window and print
      the difference between the original and the displaced image after
      correlation. `128 Gray` means no difference, lighter means positive and
      darker negative.

    kernel_file (:obj:`str`, optional): Where `crappy_install_dir` is the root
      directory of the installation of crappy (``crappy.__path__``).

    mul (:obj:`float`, optional): This parameter is critical. The direction
      will be multiplied by this scalar before being added to the solution. It
      defines how "fast" we move towards the solution. High value, fast
      convergence but risk to go past the solution and diverge (the program
      does not try to handle this and if the residual rises, iterations will
      stop immediately). Low value, probably more precise but slower and may
      require more iterations.

      After multiple tests, 3 was found to be a pretty acceptable value. Don't
      hesitate to adapt it to your case. Use ``verbose=3`` and see if the
      convergence is too slow or too fast.

  Note:
    The compared image can be given directly when querying the displacement
    as a parameter to :meth:`get_disp` or before, with :meth:`set_image`. You
    can provide it as a `np.ndarray` just like `orig`, or as a
    `pycuda.gpuarray.GPUArray`.
  """

  # Todo
  """
    This section lists all the considered improvements for this program.
    These features may NOT all be implemented in the future.
    They are sorted by priority.
    - Allow faster execution by executing the reduction only on a part
      of the images (random or chosen)
    - Add the possibility to return the value of the deformation `Exx` and
      `Eyy` in a specific point
    - Add a parameter to return values in `%`
    - Add a filter to smooth/ignore incorrect values
    - Allow a reset of the reference picture for simple deformations to
      enhance robustness in case of large deformations or lightning changes
    - Restart iterating from `0` once in a while to see if the residual is
      lower. Can be useful to recover when diverged critically due to an
      incorrect image (Shadow, obstruction, flash, camera failure, ...)
  """

  def __init__(self, img_size, **kwargs):
    global context
    if 'context' in kwargs:
      context = kwargs.pop('context')
    else:
      cuda.init()
      try:
        from pycuda.tools import make_default_context
      except (ImportError, ModuleNotFoundError):
        make_default_context = OptionalModule("pycuda",
                                              "PyCUDA and CUDA are necessary "
                                              "to use GPUCorrel")
      context = make_default_context()
    unknown = []
    for k in kwargs.keys():
      if k not in ['verbose', 'levels', 'resampling_factor', 'kernel_file',
                   'iterations', 'show_diff', 'Nfields', 'img',
                   'fields', 'mask', 'mul']:
        unknown.append(k)
    if len(unknown) != 0:
      warnings.warn("Unrecognized parameter" + (
        's: ' + str(unknown) if len(unknown) > 1 else ': ' + unknown[0]),
        SyntaxWarning)
    self.verbose = kwargs.get("verbose", 0)
    self.debug(3, "You set the verbose level to the maximum.\n\
It may help finding bugs or tracking errors but it may also \
impact the program performance as it will print A LOT of \
output and add GPU->CPU copies only to print information.\n\
If it is not desired, consider lowering the verbosity: \
1 or 2 is a reasonable choice, \
0 won't show anything except for errors.")
    self.levels = kwargs.get("levels", 5)
    self.loop = 0
    self.resamplingFactor = kwargs.get("resampling_factor", 2)
    h, w = img_size
    self.nbIter = kwargs.get("iterations", 4)
    self.debug(1, "Initializing... Master resolution:", img_size,
               "levels:", self.levels, "verbosity:", self.verbose)

    # Computing dimensions of the different levels #
    self.h, self.w = [], []
    for i in range(self.levels):
      self.h.append(int(round(h / (self.resamplingFactor ** i))))
      self.w.append(int(round(w / (self.resamplingFactor ** i))))

    if kwargs.get("Nfields") is not None:
      self.Nfields = kwargs.get("Nfields")
    else:
      try:
        self.Nfields = len(kwargs["fields"])
      except KeyError:
        self.debug(0, "Error! You must provide the number of fields at init. \
Add Nfields=x or directly set fields with fields=list/tuple")
        raise ValueError

    kernel_file = kwargs.get("kernel_file")
    if kernel_file is None:
      self.debug(3, "Kernel file not specified, using the one in crappy dir")
      from crappy import __path__ as crappy_path
      kernel_file = crappy_path[0] + "/data/kernels.cu"
    self.debug(3, "Kernel file:", kernel_file)

    # Creating a new instance of CorrelStage for each stage #
    self.correl = []
    for i in range(self.levels):
      self.correl.append(CorrelStage((self.h[i], self.w[i]),
                                     verbose=self.verbose,
                                     Nfields=self.Nfields,
                                     iterations=self.nbIter,
                                     show_diff=(i == 0 and kwargs.get(
                                         "show_diff", False)),
                                     mul=kwargs.get("mul", 3),
                                     kernel_file=kernel_file))

    # Set original image if provided #
    if kwargs.get("img") is not None:
      self.set_orig(kwargs.get("img"))

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
      self.src += s.format(i)  # Adding textures for the quick fields
      # resampling

    self.mod = SourceModule(self.src)

    self.texFx = []
    self.texFy = []
    self.resampleF = []
    for i in range(self.Nfields):
      self.texFx.append(self.mod.get_texref("texFx%d" % i))
      self.texFy.append(self.mod.get_texref("texFy%d" % i))
      self.resampleF.append(self.mod.get_function("resample%d" % i))
      self.resampleF[i].prepare("PPii", texrefs=[self.texFx[i], self.texFy[i]])

    for t in self.texFx + self.texFy:
      t.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
      t.set_filter_mode(cuda.filter_mode.LINEAR)
      t.set_address_mode(0, cuda.address_mode.BORDER)
      t.set_address_mode(1, cuda.address_mode.BORDER)

    # Set fields if provided #
    if kwargs.get("fields") is not None:
      self.set_fields(kwargs.get("fields"))

    if kwargs.get("mask") is not None:
      self.set_mask(kwargs.get("mask"))

  def get_fields(self, y=None, x=None):
    """Returns the fields, resampled to size `(y,x)`."""

    if x is None or y is None:
      y = self.h[0]
      x = self.w[0]
    out_x = gpuarray.empty((self.Nfields, y, x), np.float32)
    out_y = gpuarray.empty((self.Nfields, y, x), np.float32)
    grid = (int(ceil(x / 32)), int(ceil(y / 32)))
    block = (int(ceil(x / grid[0])), int(ceil(y / grid[1])), 1)
    for i in range(self.Nfields):
      self.resampleF[i].prepared_call(grid, block,
                                      out_x[i, :, :].gpudata,
                                      out_y[i, :, :].gpudata,
                                      np.int32(x), np.int32(y))
    return out_x, out_y

  def debug(self, n, *s):
    """To print debug info.

    First argument is the level of the message.
    It wil be displayed only if the `self.debug` is superior or equal.
    """

    if n <= self.verbose:
      print("  " * (n - 1) + "[Correl]", *s)

  def set_orig(self, img):
    """To set the original image.

    This is the reference with which the second image will be compared.
    """

    self.debug(2, "updating original image")
    assert isinstance(img, np.ndarray), "Image must be a numpy array"
    assert len(img.shape) == 2, "Image must have 2 dimensions (got {})" \
      .format(len(img.shape))
    assert img.shape == (self.h[0], self.w[0]), "Wrong size!"
    if img.dtype != np.float32:
      warnings.warn("Correl() takes arrays with dtype np.float32 \
to allow GPU computing (got {}). Converting to float32."
                    .format(img.dtype), RuntimeWarning)
      img = img.astype(np.float32)

    self.correl[0].set_orig(img)
    for i in range(1, self.levels):
      self.correl[i - 1].resample_orig(self.h[i], self.w[i],
                                      self.correl[i].devOrig)
      self.correl[i].update_orig()

  def set_fields(self, fields):
    assert self.Nfields == len(fields), \
      "Cannot change the number of fields on the go!"
    # Choosing the right function to copy
    if isinstance(fields[0], str) or isinstance(fields[0][0], np.ndarray):
      to_array = cuda.matrix_to_array
    elif isinstance(fields[0][0], gpuarray.GPUArray):
      to_array = cuda.gpuarray_to_array
    else:
      self.debug(0, "Error ! Incorrect fields argument. \
See docstring of Correl")
      raise ValueError
    # These list store the arrays for the fields texture
    # (to be interpolated quickly for each stage)
    self.fieldsXArray = []
    self.fieldsYArray = []
    for i in range(self.Nfields):
      if isinstance(fields[i], str):
        fields[i] = get_field(fields[i].lower(),
            self.h[0], self.w[0])

      self.fieldsXArray.append(to_array(fields[i][0], "C"))
      self.texFx[i].set_array(self.fieldsXArray[i])
      self.fieldsYArray.append(to_array(fields[i][1], "C"))
      self.texFy[i].set_array(self.fieldsYArray[i])
    for i in range(self.levels):
      self.correl[i].set_fields(*self.get_fields(self.h[i], self.w[i]))

  def prepare(self):
    for c in self.correl:
      c.prepare()
    self.debug(2, "Ready!")

  def save_all_images(self, name="out"):
    try:
      import cv2
    except (ModuleNotFoundError, ImportError):
      cv2 = OptionalModule("opencv-python")
    self.debug(1, "Saving all images with the name", name + "X.png")
    for i in range(self.levels):
      out = self.correl[i].devOrig.get().astype(np.uint8)
      cv2.imwrite(name + str(i) + ".png", out)

  def set_image(self, img_d):
    if img_d.dtype != np.float32:
      warnings.warn("Correl() takes arrays with dtype np.float32 \
to allow GPU computing (got {}). Converting to float32."
                    .format(img_d.dtype), RuntimeWarning)
      img_d = img_d.astype(np.float32)
    self.correl[0].set_image(img_d)
    for i in range(1, self.levels):
      self.correl[i].set_image(
        self.correl[i - 1].resample_d(self.correl[i].h, self.correl[i].w))

  def set_mask(self, mask):
    for i in range(self.levels):
      self.correl[i].set_mask(interp_nearest(mask, self.h[i], self.w[i]))

  def get_disp(self, img_d=None):
    """To get the displacement.

    This will perform the correlation routine on each stage, initializing with
    the previous values every time it will return the computed parameters
    as a list.
    """

    self.loop += 1
    if img_d is not None:
      self.set_image(img_d)
    try:
      disp = self.last / (self.resamplingFactor ** self.levels)
    except AttributeError:
      disp = np.array([0] * self.Nfields, dtype=np.float32)
    for i in reversed(range(self.levels)):
      disp *= self.resamplingFactor
      self.correl[i].set_disp(disp)
      disp = self.correl[i].get_disp()
      self.last = disp
    # Every 10 images, print the values (if debug >=2)
    if self.loop % 10 == 0:
      self.debug(2, "Loop", self.loop, ", values:", self.correl[0].devX.get(),
                 ", res:", self.correl[0].res / 1e6)
    return disp

  def get_res(self, lvl=0):
    """Returns the last residual of the specified level (`0` by default).

    Usually, the correlation is correct when `res < ~1e9-10` but it really
    depends on the images: you need to find the value that suit your own
    images, depending on the resolution, contrast, correlation method etc...
    You can use :meth:`write_diff_file` to visualize the difference between the
    two images after correlation.
    """

    return self.correl[lvl].res

  def write_diff_file(self, level=0):
    """To see the difference between the two images with the computed
    parameters.

    It writes a single channel picture named `"diff.png"` where `128 gray` is
    exact equality, lighter pixels show positive difference and darker pixels
    a negative difference. Useful to see if correlation succeeded and to
    identify the origin of non convergence.
    """

    self.correl[level].write_diff_file()

  @staticmethod
  def clean():
    """Needs to be called at the end, to destroy the context properly."""

    context.pop()
