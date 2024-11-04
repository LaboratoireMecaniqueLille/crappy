# coding:utf-8

import warnings
from math import ceil
import numpy as np
from pkg_resources import resource_filename
from typing import Any, Optional, Union, Literal
from pathlib import Path
from itertools import chain
import logging

from .fields import get_field
from ..._global import OptionalModule
try:
  import pycuda.tools
  import pycuda.driver
  import pycuda.compiler
  import pycuda.gpuarray as gpuarray
  import pycuda.reduction
except (ImportError, ModuleNotFoundError):
  pycuda = OptionalModule("pycuda",
                          "PyCUDA and CUDA are necessary to use GPUCorrel")
  gpuarray = OptionalModule("pycuda",
                            "PyCUDA and CUDA are necessary to use GPUCorrel")


def interp_nearest(arr: np.ndarray, ny: int, nx: int) -> np.ndarray:
  """Reshapes an input array to the specified dimension using the nearest
  interpolation, using only :mod:`numpy` methods.

  Args:
    arr: The array to interpolate.
    ny: The new dimension along the `y` axis.
    nx: The new dimension along the `y` axis.

  Returns:
    A reshaped version of the input array obtained with the nearest
    interpolation.
    
  .. versionadded:: 1.4.0
  """

  # If the shape is already fine, nothing more to do
  if arr.shape == (ny, nx):
    return arr

  y, x = arr.shape
  rx, ry = x / nx, y / ny
  out = np.empty((ny, nx), dtype=np.float32)

  # Filling individually each value of the returned array
  for j in range(ny):
    for i in range(nx):
      out[j, i] = arr[int(ry * j + .5), int(rx * i + .5)]

  return out


class CorrelStage:
  """Represents a stage of the pyramid used by the :class:`GPUCorrelTool` for
  performing GPU correlation.

  This class actually performs the GPU computation, while the calling classes
  only manage the data.

  .. versionadded:: 1.4.0
  """

  def __init__(self,
               img_size: tuple[int, int],
               logger_name: str,
               verbose: int = 0,
               iterations: int = 5,
               mul: float = 3,
               n_fields: Optional[int] = None,
               kernel_file: Optional[Union[Path, str]] = None) -> None:
    """Sets the args and instantiates the :mod:`pycuda` objects.

    Args:
      img_size: The shape of the images to process. It is given beforehand so
        that the memory can be allocated before the test starts.
      logger_name: The name of the parent logger, as a :obj:`str`.

        .. versionadded:: 2.0.0
      verbose: The verbose level as an integer, between `0` and `3`. At level
        `0` no information is displayed, and at level `3` so much information
        is displayed that is slows the code down.
      iterations: The maximum number of iterations to run before returning the
        results. The results may be returned before if the residuals start
        increasing.
      mul: The scalar by which the direction will be multiplied before being
        added to the solution. If it's too high, the convergence will be fast
        but there's a risk that to go past the solution and to diverge. If it's
        too low, the convergence will be slower and require more iterations.
        `3` was found to be an acceptable value in most cases, but it is
        recommended to tune this value for each application so that the
        convergence is neither too slow nor too fast.
      n_fields: The number of fields to project the displacement on.

        .. versionchanged:: 1.5.10 renamed from *Nfields* to *n_fields*
      kernel_file: The path to the file containing the kernels to use for the
        correlation. Can be a :obj:`pathlib.Path` object or a :obj:`str`.

    .. versionchanged:: 1.5.10
       now explicitly listing the *verbose*, *iterations*, *mul* and
       *kernel_file* arguments
    .. versionremoved:: 1.5.10 *show_diff*, *img*, *mask* and *fields* arguments
    """

    self._logger: Optional[logging.Logger] = None
    self._logger_name = logger_name

    # Setting the args
    self._verbose = verbose
    self._iterations = iterations
    self._mul = mul
    self._n_fields = n_fields
    self._ready = False
    self._height, self._width = img_size
    self._debug(2, f"Initializing with resolution {img_size}")

    # These attributes will be set later
    self._r_x, self._r_y = None, None
    self._mask_array = None
    self._array = None
    self._array_d = None
    self._fields = False
    self._r_grid = None
    self._r_block = None
    self._dev_r_out = None
    self.res = None

    # Setting the grid and the block for the kernel
    self._grid = (int(ceil(self._width / 32)), int(ceil(self._height / 32)))
    self._block = (int(ceil(self._width / self._grid[0])),
                   int(ceil(self._height / self._grid[1])), 1)
    self._debug(3, f"Default grid: {self._grid}, block: {self._block}")

    # dev_g stores the G arrays (to compute the research direction)
    self._dev_g = [gpuarray.empty(img_size, np.float32)
                   for _ in range(self._n_fields)]
    # dev_fields_x/y store the fields value along X and Y
    self._dev_fields_x = [gpuarray.empty(img_size, np.float32)
                          for _ in range(self._n_fields)]
    self._dev_fields_y = [gpuarray.empty(img_size, np.float32)
                          for _ in range(self._n_fields)]
    # dev_h Stores the Hessian matrix
    self._dev_h = np.zeros((self._n_fields, self._n_fields), np.float32)
    # And dev_h_i stores its invert
    self.dev_h_i = gpuarray.empty((self._n_fields, self._n_fields), np.float32)
    # dev_out is written with the difference of the images
    self._dev_out = gpuarray.empty((self._height, self._width), np.float32)
    # dev_x stores the value of the parameters (what is actually computed)
    self.dev_x = gpuarray.empty(self._n_fields, np.float32)
    # dev_vec stores the research direction
    self._dev_vec = gpuarray.empty(self._n_fields, np.float32)
    # dev_orig stores the original image on the device
    self.dev_orig = gpuarray.empty(img_size, np.float32)
    # dev_grad_x store the gradient along X of the original image on the device
    self._dev_grad_x = gpuarray.empty(img_size, np.float32)
    # And along Y
    self._dev_grad_y = gpuarray.empty(img_size, np.float32)

    # Opening the kernel file and compiling the module
    if kernel_file is None:
      self._debug(2, "Kernel file not specified")
      kernel_file = resource_filename('crappy',
                                      'tool/image_processing/kernels.cu')
    with open(kernel_file, "r") as file:
      self._debug(3, "Sourcing module")
      mod = pycuda.compiler.SourceModule(file.read() % (self._width,
                                                        self._height,
                                                        self._n_fields))

    # Extracting individual functions from the data of the kernel file
    self._resample_orig_krnl = mod.get_function('resampleO')
    self._resample_krnl = mod.get_function('resample')
    self._gradient_krnl = mod.get_function('gradient')
    self._make_g_krnl = mod.get_function('makeG')
    self._make_diff = mod.get_function('makeDiff')
    self._dot_krnl = mod.get_function('myDot')
    self._add_krnl = mod.get_function('kadd')
    self._mul_red_krnl = pycuda.reduction.ReductionKernel(
      np.float32, neutral="0", reduce_expr="a+b", map_expr="x[i]*y[i]",
      arguments="float *x, float *y")
    self._least_square = pycuda.reduction.ReductionKernel(
      np.float32, neutral="0", reduce_expr="a+b", map_expr="x[i]*x[i]",
      arguments="float *x")

    # Getting the texture references
    self._tex = mod.get_texref('tex')
    self._tex_d = mod.get_texref('tex_d')
    self._tex_mask = mod.get_texref('texMask')

    # All textures use normalized coordinates except for the mask
    for t in (self._tex, self._tex_d):
      t.set_flags(pycuda.driver.TRSF_NORMALIZED_COORDINATES)
    for t in (self._tex, self._tex_d, self._tex_mask):
      t.set_filter_mode(pycuda.driver.filter_mode.LINEAR)
      t.set_address_mode(0, pycuda.driver.address_mode.BORDER)
      t.set_address_mode(1, pycuda.driver.address_mode.BORDER)

    # Preparing kernels for less overhead when called
    self._resample_orig_krnl.prepare("Pii", texrefs=[self._tex])
    self._resample_krnl.prepare("Pii", texrefs=[self._tex_d])
    self._gradient_krnl.prepare("PP", texrefs=[self._tex])
    self._make_diff.prepare("PPPP",
                            texrefs=[self._tex, self._tex_d, self._tex_mask])
    self._add_krnl.prepare("PfP")

  def set_orig(self, img) -> None:
    """Sets the original image from a given CPU or GPU array"""

    if isinstance(img, np.ndarray):
      self._debug(3, "Setting original image from ndarray")
      self.dev_orig.set(img)

    elif isinstance(img, gpuarray.GPUArray):
      self._debug(3, "Setting original image from GPUArray")
      self.dev_orig = img

    else:
      raise ValueError("Error ! Unknown type of data given to set_orig()")

    self.update_orig()

  def update_orig(self) -> None:
    """Updates the original image."""

    self._debug(3, "Updating original image")

    self._array = pycuda.driver.gpuarray_to_array(self.dev_orig, 'C')
    self._tex.set_array(self._array)
    self._compute_gradients()
    self._ready = False

  def prepare(self) -> None:
    """Computes all the necessary tables to perform correlation."""

    # Sets the mask array if none was specified
    if self._mask_array is None:
      self._debug(2, "No mask set when preparing, using a basic one, with a "
                     "border of 5% the dimension")
      mask = np.zeros((self._height, self._width), np.float32)
      mask[self._height // 20: -self._height // 20,
           self._width // 20: -self._width // 20] = 1
      self.set_mask(mask)

    # Only necessary to prepare if not already ready
    if not self._ready:
      # Checking that everything's set for preparing
      if self._array is None:
        self._debug(1, "Tried to prepare but original texture is not set !")
      elif not self._fields:
        self._debug(1, "Tried to prepare but fields are not set !")

      # Actually computing the tables
      else:
        self._make_g()
        self._make_h()
        self._ready = True
        self._debug(3, "Ready!")

    else:
      self._debug(1, "Tried to prepare when unnecessary, doing nothing...")

  def resample_orig(self, new_y: int, new_x: int, dev_out) -> None:
    """Resamples the original image with a new dimension."""

    # New grid and block to be used
    grid = (int(ceil(new_x / 32)), int(ceil(new_y / 32)))
    block = (int(ceil(new_x / grid[0])), int(ceil(new_y / grid[1])), 1)
    self._debug(3, f"Resampling Orig texture, grid: {grid}, block: {block}")

    # Now resampling
    self._resample_orig_krnl.prepared_call(self._grid, self._block,
                                           dev_out.gpudata,
                                           np.int32(new_x), np.int32(new_y))
    self._debug(3, f"Resampled original texture to {dev_out.shape}")

  def resample_d(self, new_y: int, new_x: int) -> Any:
    """Resamples the texture with a new dimension and returns it in a GPU
    array."""

    if (self._r_x, self._r_y) != (np.int32(new_x), np.int32(new_y)):
      self._r_grid = (int(ceil(new_x / 32)), int(ceil(new_y / 32)))
      self._r_block = (int(ceil(new_x / self._r_grid[0])),
                       int(ceil(new_y / self._r_grid[1])), 1)
      self._r_x, self._r_y = np.int32(new_x), np.int32(new_y)
      self._dev_r_out = gpuarray.empty((new_y, new_x), np.float32)

    self._debug(3, f"Resampling img_d texture to {new_y, new_x}, "
                   f"grid: {self._r_grid}, block:{self._r_block}")

    self._resample_krnl.prepared_call(self._r_grid, self._r_block,
                                      self._dev_r_out.gpudata,
                                      self._r_x, self._r_y)
    return self._dev_r_out

  def set_fields(self, fields_x, fields_y) -> None:
    """Sets the fields on which hto project the displacement."""

    self._debug(2, "Setting fields")

    if isinstance(fields_x, np.ndarray):
      self._dev_fields_x.set(fields_x)
      self._dev_fields_y.set(fields_y)

    elif isinstance(fields_x, gpuarray.GPUArray):
      self._dev_fields_x = fields_x
      self._dev_fields_y = fields_y

    self._fields = True

  def set_image(self, img_d) -> None:
    """Set the current image, to be compared with the reference image."""

    if isinstance(img_d, np.ndarray):
      self._debug(3, "Creating texture from numpy array")
      self._array_d = pycuda.driver.matrix_to_array(img_d, "C")

    elif isinstance(img_d, gpuarray.GPUArray):
      self._debug(3, "Creating texture from gpuarray")
      self._array_d = pycuda.driver.gpuarray_to_array(img_d, "C")

    else:
      self._debug(0, "Error ! Unknown type of data given to .set_image()")
      raise ValueError

    self._tex_d.set_array(self._array_d)
    self.dev_x.set(np.zeros(self._n_fields, dtype=np.float32))

  def set_mask(self, mask) -> None:
    """Sets the mask to use for weighting the images."""

    self._debug(3, "Setting the mask")

    if not mask.dtype == np.float32:
      self._debug(2, "Converting the mask to float32")
      mask = mask.astype(np.float32)

    if isinstance(mask, np.ndarray):
      self.maskArray = pycuda.driver.matrix_to_array(mask, 'C')

    elif isinstance(mask, gpuarray.GPUArray):
      self.maskArray = pycuda.driver.gpuarray_to_array(mask, 'C')

    else:
      raise ValueError("Error! Mask data type not understood")

    self._tex_mask.set_array(self.maskArray)

  def set_disp(self, x) -> None:
    """Sets the displacement fields computed from the previous stages."""

    if isinstance(x, gpuarray.GPUArray):
      self.dev_x = x

    elif isinstance(x, np.ndarray):
      self.dev_x.set(x)

    else:
      raise ValueError("Error! Unknown type of data given to "
                       "CorrelStage.set_disp")

  def get_disp(self, img_d=None) -> Any:
    """Projects the displacement on the base fields, and returns the result."""

    self._debug(3, "Calling main routine")

    if not self._ready:
      self._debug(2, "Wasn't ready ! Preparing...")
      self.prepare()

    if img_d is not None:
      self.set_image(img_d)

    if self._array_d is None:
      raise ValueError("Did not set the image, use set_image() before calling "
                       "get_disp !")

    self._debug(3, "Computing first diff table")

    self._make_diff.prepared_call(self._grid, self._block,
                                  self._dev_out.gpudata,
                                  self.dev_x.gpudata,
                                  self._dev_fields_x.gpudata,
                                  self._dev_fields_y.gpudata)

    self.res = self._least_square(self._dev_out).get()
    self._debug(3, f"res: {self.res / 1e6}")

    for i in range(self._iterations):
      self._debug(3, f"Iteration {i}")

      # Computing the direction of the gradient of each parameter
      for j in range(self._n_fields):
        self._dev_vec[j] = self._mul_red_krnl(self._dev_g[j], self._dev_out)

      # Newton method: the gradient vector is multiplied by the pre-inverted
      # Hessian, self._dev_vec now contains the actual research direction
      self._dot_krnl(self.dev_h_i, self._dev_vec,
                     grid=(1, 1), block=(self._n_fields, 1, 1))

      # This line simply adds mul times the research direction to self._dev_x
      # with a really simple kernel (does self._dev_x += mul * self._dev_vec)
      self._add_krnl.prepared_call((1, 1), (self._n_fields, 1, 1),
                                   self.dev_x.gpudata, self._mul,
                                   self._dev_vec.gpudata)

      # Simple way to prevent unnecessary copy of data
      if self._verbose >= 3:
        self._debug(3, f"Direction: {self._dev_vec.get()}")
        self._debug(3, f"New X: {self.dev_x.get()}")

      # Getting the new residuals
      self._make_diff.prepared_call(self._grid, self._block,
                                    self._dev_out.gpudata,
                                    self.dev_x.gpudata,
                                    self._dev_fields_x.gpudata,
                                    self._dev_fields_y.gpudata)

      prev_res = self.res
      self.res = self._least_square(self._dev_out).get()

      # Handling the case when the residuals start increasing
      if self.res >= prev_res:
        self._debug(3, f"Diverting from the solution "
                       f"new res={self.res / 1e6} >= {prev_res / 1e6}!")

        self._add_krnl.prepared_call((1, 1), (self._n_fields, 1, 1),
                                     self.dev_x.gpudata,
                                     -self._mul,
                                     self._dev_vec.gpudata)
        self.res = prev_res
        self._debug(3, f"Undone: X={self.dev_x.get()}")
        break

      self._debug(3, f"res: {self.res / 1e6}")

    return self.dev_x.get()

  def get_data_display(self) -> np.ndarray:
    """Returns the necessary data for displaying the difference between the
    reference and current image."""

    return (self._dev_out.get() + 128).astype(np.uint8)

  def _make_g(self) -> None:
    """Computes the tables."""

    for i in range(self._n_fields):
      self._make_g_krnl(self._dev_g[i].gpudata, self._dev_grad_x.gpudata,
                        self._dev_grad_y.gpudata,
                        self._dev_fields_x[i], self._dev_fields_y[i],
                        block=self._block, grid=self._grid)

  def _make_h(self) -> None:
    """Computes the tables."""

    for i in range(self._n_fields):
      for j in range(i + 1):
        self._dev_h[i, j] = self._mul_red_krnl(self._dev_g[i],
                                               self._dev_g[j]).get()
        if i != j:
          self._dev_h[j, i] = self._dev_h[i, j]

    self._debug(3, f"Hessian: {self._dev_h}")
    self.dev_h_i.set(np.linalg.inv(self._dev_h))

    if self._verbose >= 3:
      self._debug(3, f"Inverted Hessian: {self.dev_h_i.get()}")

  def _debug(self, level: int, msg: str) -> None:
    """Displays the provided debug message only if its debug level is lower
    than or equal to the verbose level."""

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{self._logger_name}.{type(self).__name__}")

    if level <= self._verbose:
      self._logger.log(logging.INFO, msg)

  def _compute_gradients(self) -> None:
    """Wrapper to call the gradient kernel."""

    self._gradient_krnl.prepared_call(self._grid, self._block,
                                      self._dev_grad_x.gpudata,
                                      self._dev_grad_y.gpudata)


class GPUCorrelTool:
  """This class is the core of the :class:`~crappy.blocks.GPUCorrel` and 
  :class:`~crappy.blocks.GPUVE` Blocks.

  It receives images from a :class:`~crappy.camera.Camera`, and performs 
  GPU-accelerated image correlation on each received image. From this 
  correlation, rigid body displacements or other fields are identified.

  This class  is meant to be efficient enough to run in real-time. It relies on
  the :class:`~crappy.tool.image_processing.gpu_correl.CorrelStage` class (not
  documented) to perform correlation on different scales. It mainly takes a
  list of base fields and a reference image as inputs, and project the
  displacement between the current image and the reference one on the base of
  fields. The optimal fit is achieved by lowering the residuals with a
  least-squares method.

  The projection on the base is performed sequentially, using the results
  obtained at stages with low resolution to initialize the computation on
  stages with higher resolution. A Newton method is used to converge towards
  an optimal solution.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *GPUCorrel* to *GPUCorrelTool*
  """

  context = None

  def __init__(self,
               logger_name: str,
               context: Optional[Any] = None,
               verbose: int = 0,
               levels: int = 5,
               resampling_factor: float = 2,
               kernel_file: Optional[Union[str, Path]] = None,
               iterations: int = 4,
               fields: Optional[list[Union[Literal['x', 'y', 'r', 'exx', 'eyy',
                                                   'exy', 'eyx', 'exy2', 'z'],
                                           np.ndarray]]] = None,
               ref_img: Optional[np.ndarray] = None,
               mask: Optional[np.ndarray] = None,
               mul: float = 3) -> None:
    """Sets the args and a few parameters of :mod:`pycuda`.

    Args:
      logger_name: The name of the parent :obj:`~logging.Logger`, to be used
        for setting the Logger of the class.

        .. versionadded:: 2.0.0
      context: Optionally, the :mod:`pycuda` context to use. If not specified,
        a new context is instantiated.

        .. versionadded:: 1.5.10
      verbose: The verbose level as an integer, between `0` and `3`. At level
        `0` no information is displayed, and at level `3` so much information
        is displayed that it slows the code down.
      levels: Number of levels of the pyramid. More levels may help converging
        on images with large strain, but may fail on images that don't contain
        low spatial frequency. Fewer levels mean that the program runs faster.
      resampling_factor: The factor by which the resolution is divided between
        each stage of the pyramid. A low factor ensures coherence between the
        stages, but is more computationally intensive. A high factor allows
        reaching a finer detail level, but may lead to a coherence loss between
        the stages.
      kernel_file: The path to the file containing the kernels to use for the
        correlation. Can be a :obj:`pathlib.Path` object or a :obj:`str`. If
        not provided, the default :ref:`GPU Kernels` are used.
      iterations: The maximum number of iterations to run before returning the
        results. The results may be returned before if the residuals start
        increasing.
      fields: The base of fields to use for the projection, given as a
        :obj:`list` of :obj:`str` or :mod:`numpy` arrays (both types can be
        mixed). Strings are for using automatically-generated fields, the
        available ones are :
        ::

          'x', 'y', 'r', 'exx', 'eyy', 'exy', 'eyx', 'exy2', 'z'

        If users provide their own fields as arrays, they will be used as-is to
        run the correlation. The user-provided fields must be of shape:
        ::

          (patch_height, patch_width, 2)

        .. versionchanged:: 2.0.5 provided fields can now be numpy arrays
      ref_img: The reference image, as a 2D :obj:`numpy.array` with `dtype`
        `float32`. It can either be given at :meth:`__init__`, or set later
        with :meth:`set_orig`.

        .. versionchanged:: 1.5.10 renamed from *img* to *ref_img*
      mask: The mask used for weighting the region of interest on the image. It
        is generally used to prevent unexpected behavior on the border of the
        image.
      mul: The scalar by which the direction will be multiplied before being
        added to the solution. If it's too high, the convergence will be fast
        but there's a risk to go past the solution and to diverge. If it's too
        low, the convergence will be slower and require more iterations. `3`
        was found to be an acceptable value in most cases, but it is
        recommended to tune this value for each application so that the
        convergence is neither too slow nor too fast.
    
    .. versionchanged:: 1.5.10 
       now explicitly listing the *verbose*, *levels*, *resampling_factor*,
       *kernel_file*, *iterations*, *fields*, *mask* and *mul* arguments
    .. versionremoved:: 1.5.10 *show_diff* and *img_size* arguments
    """

    self._context = context
    self._logger: Optional[logging.Logger] = None
    self._logger_name = logger_name

    self._verbose = verbose
    self._debug(3, "You set the verbose level to the maximum.\nIt may help "
                   "finding bugs or tracking errors but it may also impact the"
                   " program performance as it will display A LOT of output "
                   "and add GPU->CPU copies only to display information.\nIf "
                   "it is not desired, consider lowering the verbosity: 1 or "
                   "2 is a reasonable choice, 0 won't show anything except "
                   "errors.")

    # Setting the args
    self._levels = levels
    self._resampling_factor = resampling_factor
    self._iterations = iterations
    self._mul = mul
    self._fields = fields
    self._n_fields = len(fields)
    self._mask = mask
    self._ref_img = ref_img
    self._loops = 0
    self._last = None

    # These attributes will be set later
    self._heights, self._widths = None, None
    self._stages = None
    self._tex_fx = None
    self._tex_fy = None
    self._resample = None

    self._debug(1, f"Initializing... levels: {levels}, verbosity: {verbose}")

    # Getting the kernel file
    if kernel_file is None:
      self._debug(3, "Kernel file not specified, using the default one of "
                     "crappy")
      self._kernel_file = resource_filename('crappy',
                                            'tool/image_processing/kernels.cu')
    else:
      self._kernel_file = kernel_file
    self._debug(3, f"Kernel file:{self._kernel_file}")

  def set_img_size(self, img_size: tuple[int, int]) -> None:
    """Sets the image shape, and calls the methods that need this information
    for running."""

    self._debug(1, f"Setting master resolution: {img_size},")

    # Initializing the pycuda context
    if self._context is not None:
      GPUCorrelTool.context = self._context
    else:
      pycuda.driver.init()
      GPUCorrelTool.context = pycuda.tools.make_default_context()

    src_txt = """
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

    # Setting the textures for a faster resampling
    src = ''.join([src_txt.format(i) for i in range(self._n_fields)])
    source_module = pycuda.compiler.SourceModule(src)

    self._tex_fx = [source_module.get_texref(f"texFx{i}")
                    for i in range(self._n_fields)]
    self._tex_fy = [source_module.get_texref(f"texFy{i}")
                    for i in range(self._n_fields)]
    self._resample = [source_module.get_function(f"resample{i}")
                      for i in range(self._n_fields)]

    for tex_fx, tex_fy, resample in zip(self._tex_fx, self._tex_fy,
                                        self._resample):
      resample.prepare('PPii', texrefs=[tex_fx, tex_fy])

    for tex in chain(self._tex_fx, self._tex_fy):
      tex.set_flags(pycuda.driver.TRSF_NORMALIZED_COORDINATES)
      tex.set_filter_mode(pycuda.driver.filter_mode.LINEAR)
      tex.set_address_mode(0, pycuda.driver.address_mode.BORDER)
      tex.set_address_mode(1, pycuda.driver.address_mode.BORDER)

    # Setting the dimensions for each stage
    height, width, *_ = img_size
    self._heights = [round(height / (self._resampling_factor ** i))
                     for i in range(self._levels)]
    self._widths = [round(width / (self._resampling_factor ** i))
                    for i in range(self._levels)]

    # Initializing all the stages
    self._stages = [CorrelStage(img_size=(height, width),
                                logger_name=f"{self._logger_name}."
                                            f"{type(self).__name__}",
                                verbose=self._verbose,
                                n_fields=self._n_fields,
                                iterations=self._iterations,
                                mul=self._mul,
                                kernel_file=self._kernel_file)
                    for i, (height, width) in enumerate(zip(self._heights,
                                                            self._widths))]

    # Now that the stages exist, setting the reference image, fields, and mask
    if self._ref_img is not None:
      self.set_orig(self._ref_img)

    if self._fields is not None:
      self._set_fields(self._fields)

    if self._mask is not None:
      self._set_mask(self._mask)

  def set_orig(self, img: np.ndarray) -> None:
    """Sets the reference image, to which the following images will be
    compared."""

    self._debug(2, "Updating the original image")

    # Casting to float32 if needed
    if img.dtype != np.float32:
      warnings.warn("Correl() takes arrays with dtype np.float32 to allow GPU "
                    "computing (got {}). Converting to float32."
                    .format(img.dtype), RuntimeWarning)
      img = img.astype(np.float32)

    # Setting the reference image for all stages
    self._stages[0].set_orig(img)
    for prev_stage, stage, height, width in zip(self._stages[:-1],
                                                self._stages[1:],
                                                self._heights, self._widths):
      prev_stage.resample_orig(height, width, stage.dev_orig)
      stage.update_orig()

  def prepare(self) -> None:
    """Prepares all the stages before starting the test."""

    for stage in self._stages:
      stage.prepare()

    self._debug(2, "Ready !")

  def get_disp(self, img_d: Optional[np.ndarray] = None) -> Any:
    """To get the displacement.

    This will perform the correlation routine on each stage, initializing with
    the previous values every time it will return the computed parameters
    as a list.
    """

    if img_d is not None:
      self._set_image(img_d)

    if self._last is not None:
      disp = self._last / (self._resampling_factor ** self._levels)
    else:
      disp = np.array([0] * self._n_fields, dtype=np.float32)

    for stage in reversed(self._stages):
      disp *= self._resampling_factor
      stage.set_disp(disp)
      disp = stage.get_disp()

    self._loops += 1
    if not self._loops % 10:
      self._debug(2,
                  f"Loop {self._loops}, values: {self._stages[0].dev_x.get()}"
                  f", res: {self._stages[0].res / 1e6}")

    return disp

  def get_data_display(self) -> np.ndarray:
    """Returns the necessary data for displaying the difference between the
    reference and current image."""

    return self._stages[0].get_data_display()

  def get_res(self, lvl: int = 0):
    """Returns the last residual of the specified level."""

    return self._stages[lvl].res

  @staticmethod
  def clean():
    """Needs to be called at the end, to destroy the :mod:`pycuda` context
    properly."""

    GPUCorrelTool.context.pop()

  def _get_fields(self,
                  y: Optional[int] = None,
                  x: Optional[int] = None) -> (Any, Any):
    """Returns the fields, resampled to size `(y, x)`."""

    if x is None or y is None:
      y, x = self._heights[0], self._widths[0]

    out_x = gpuarray.empty((self._n_fields, y, x), np.float32)
    out_y = gpuarray.empty((self._n_fields, y, x), np.float32)
    grid = (int(ceil(x / 32)), int(ceil(y / 32)))
    block = (int(ceil(x / grid[0])), int(ceil(y / grid[1])), 1)

    for i, resample in enumerate(self._resample):
      resample.prepared_call(grid, block,
                             out_x[i, :, :].gpudata,
                             out_y[i, :, :].gpudata,
                             np.int32(x), np.int32(y))

    return out_x, out_y

  def _debug(self, level: int, msg: str) -> None:
    """Displays the provided debug message only if its debug level is lower
    than or equal to the verbose level."""

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{self._logger_name}.{type(self).__name__}")

    if level <= self._verbose:
      self._logger.log(logging.INFO, msg)

  def _set_fields(self, fields: list[Union[Literal['x', 'y', 'r', 'exx', 'eyy',
                                                   'exy', 'eyx', 'exy2', 'z'],
                                           np.ndarray]]) -> None:
    """Computes the fields based on the provided field strings, and sets them
    for each stage."""

    for field, tex_fx, tex_fy in zip(fields, self._tex_fx, self._tex_fy):

      # Getting the fields as numpy arrays
      if isinstance(field, str):
        field_x, field_y = get_field(field, self._heights[0], self._widths[0])
      elif isinstance(field, np.ndarray):
        field_x, field_y = field[:, :, 0], field[:, :, 1]
      else:
        raise TypeError("The provided fields should either be strings or "
                        "numpy arrays !")

      tex_fx.set_array(pycuda.driver.matrix_to_array(field_x, 'C'))
      tex_fy.set_array(pycuda.driver.matrix_to_array(field_y, 'C'))

    # Setting the fields for each stage
    for stage, height, width in zip(self._stages, self._heights, self._widths):
      stage.set_fields(*self._get_fields(height, width))

  def _set_image(self, img_d: np.ndarray) -> None:
    """Sets the current image for all the stages, to be compared with the
    reference image."""

    # Casting to float32 if needed
    if img_d.dtype != np.float32:
      warnings.warn(f"Correl() takes arrays with dtype np.float32 to allow GPU"
                    f" computing (got {img_d.dtype}). Converting to float32.",
                    RuntimeWarning)
      img_d = img_d.astype(np.float32)

    # Setting the current image for all stages
    self._stages[0].set_image(img_d)
    for prev_stage, stage, height, width in zip(self._stages[:-1],
                                                self._stages[1:],
                                                self._heights[1:],
                                                self._widths[1:]):
      stage.set_image(prev_stage.resample_d(height, width))

  def _set_mask(self, mask: np.ndarray) -> None:
    """Sets for each field the mask for weighting the images to process."""

    for stage, height, width in zip(self._stages, self._heights, self._widths):
      stage.set_mask(interp_nearest(mask, height, width))
