import mujoco
import numpy as np
import cv2
import threading
import atexit


class Context:
  """A context handler for GL and mujoco contexts.
  Initialises one GL and mujoco context which can be shared by multiple `Camera`s.
  """

  def __init__(self, model, max_resolution):
    """Initializes a new `Context`.
    Args:
      model: Instance of a mujoco model.
      max_resolution: Maximum resolution that cameras with this context can use.
    """

    self._max_resolution = max_resolution
    self._closed = False

    # Not really sure if we need this
    self._contexts_lock = threading.Lock()

    with self._contexts_lock:

      # Set simulation-wide max resolution
      model.vis.global_.offwidth = max_resolution[0]
      model.vis.global_.offheight = max_resolution[1]

      # Create the OpenGL context.
      self._gl = mujoco.GLContext(max_resolution[1], max_resolution[0])
      self._gl.make_current()

      # Create the MuJoCo context.
      self.mujoco = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
      mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.mujoco)

    # Can't leave de-allocation for destructors, will raise errors/warnings
    atexit.register(self.close)

  def make_current(self):
    if not self._closed and self._gl is not None:
      self._gl.make_current()
      mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.mujoco)

  def close(self):
    if self._closed:
      return
    self._closed = True
    atexit.unregister(self.close)
    with self._contexts_lock:
      if getattr(self, "_gl", None) is not None:
        self._gl.make_current()
        if getattr(self, "mujoco", None) is not None and hasattr(self.mujoco, "free"):
          self.mujoco.free()
        self._gl.free()
        self.mujoco = None
        self._gl = None


class Camera:
  """Mujoco scene camera.
  Adapted from https://github.com/deepmind/dm_control/blob/main/dm_control/mujoco/engine.py.
  Holds rendering properties such as the width and height of the viewport. The
  camera position and rotation is defined by the Mujoco camera corresponding to
  the `camera_id`. Multiple `Camera` instances may exist for a single
  `camera_id`, for example to render the same view at different resolutions.
  """

  def __init__(self,
               context,
               model,
               data,
               resolution=None,
               rgb=True,
               depth=False,
               camera_id=-1,
               maxgeom=1000,
               dt=None):
    """Initializes a new `Camera`.
    Args:
      context: Instance of a `Context`.
      model: Instance of a mujoco model class.
      data: Instance of a mujoco data class.
      resolution: An array/list of shape (Image width x image height). If none,
        global max resolution will be used.
      rgb: A boolean indicating whether an rgb array is rendered and returned.
        True by default.
      depth: A boolean indicating whether a depth array is rendered and returned.
        False by default.
      camera_id: Optional camera name or index. Defaults to -1, the free
        camera, which is always defined. A nonnegative integer or string
        corresponds to a fixed camera, which must be defined in the model XML.
        If `camera_id` is a string then the camera must also be named.
      maxgeom: Optional integer specifying the maximum number of geoms that can
        be rendered in the same scene. Default 1000.
      dt: Time between steps, optional. Required for recording videos.
    Raises:
      ValueError: If `camera_id` is outside the valid range, or if `width` or
        `height` exceed the dimensions of MuJoCo's offscreen framebuffer.
    """
    buffer_width = model.vis.global_.offwidth
    buffer_height = model.vis.global_.offheight
    if resolution is None: resolution = [buffer_width, buffer_height]
    if resolution[0] > buffer_width:
      raise ValueError('Image width {} > framebuffer width {}. Either reduce '
                       'the image width or specify a larger offscreen '
                       'framebuffer in the config\'s "run_parameters" using '
                       ' keyword "max_resolution"'.format(resolution[0], buffer_width))
    if resolution[1] > buffer_height:
      raise ValueError('Image height {} > framebuffer height {}. Either reduce '
                       'the image height or specify a larger offscreen '
                       'framebuffer in the config\'s "run_parameters" using '
                       ' keyword "max_resolution"'.format(resolution[1], buffer_height))
    if isinstance(camera_id, str):
      camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
    if camera_id < -1:
      raise ValueError('camera_id cannot be smaller than -1.')
    if camera_id >= model.ncam:
      raise ValueError('model has {} fixed cameras. camera_id={} is invalid.'.
                       format(model.ncam, camera_id))

    self._resolution = resolution
    self._model = model
    self._data = data
    self._context = context
    self._rgb = rgb
    self._depth = depth

    # Variables corresponding to structs needed by Mujoco's rendering functions.
    self._scene = mujoco.MjvScene(model=model, maxgeom=maxgeom)
    self._scene_option = mujoco.MjvOption()

    self._perturb = mujoco.MjvPerturb()
    self._perturb.active = 0
    self._perturb.select = 0

    self._rect = mujoco.MjrRect(0, 0, *self._resolution)

    self._render_camera = mujoco.MjvCamera()
    self._render_camera.fixedcamid = camera_id

    if camera_id == -1:
      self._render_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    else:
      # As defined in the Mujoco documentation, mjCAMERA_FIXED refers to a
      # camera explicitly defined in the model.
      self._render_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED

    # Internal buffers.
    self._rgb_buffer = np.empty((self._resolution[1], self._resolution[0], 3), dtype=np.uint8) if rgb else None
    self._depth_buffer = np.empty((self._resolution[1], self._resolution[0]), dtype=np.float32) if depth else None

    # Set frames per second
    self.set_fps(dt) if dt else None

  @property
  def width(self):
    """Returns the image width (number of pixels)."""
    return self._resolution[0]

  @property
  def height(self):
    """Returns the image height (number of pixels)."""
    return self._resolution[1]

  @property
  def option(self):
    """Returns the camera's visualization options."""
    return self._scene_option

  @property
  def scene(self):
    """Returns the `mujoco.MjvScene` instance used by the camera."""
    return self._scene

  def update(self, scene_option=None):
    """Updates geometry used for rendering.
    Args:
      scene_option: A custom `wrapper.MjvOption` instance to use to render
        the scene instead of the default.  If None, will use the default.
    """
    scene_option = scene_option or self._scene_option
    mujoco.mjv_updateScene(self._model, self._data,
                           scene_option, self._perturb,
                           self._render_camera, mujoco.mjtCatBit.mjCAT_ALL,
                           self._scene)

  def _render_on_gl_thread(self):
    """Performs only those rendering calls that require an OpenGL context."""

    self._context.make_current()

    # Render the scene.
    mujoco.mjr_render(self._rect, self._scene,
                      self._context.mujoco)

    # Read the contents of either the RGB or depth buffer.
    mujoco.mjr_readPixels(self._rgb_buffer if self._rgb else None,
                          self._depth_buffer if self._depth else None,
                          self._rect, self._context.mujoco)

  def render(
      self,
      scene_option=None,
  ):
    """Renders the camera view as a numpy array of pixel values.
    Args:
      scene_option: A custom `mujoco.MjvOption` instance to use to render
        the scene instead of the default.  If None, will use the default.
    Returns:
      The rendered scene, including
        * A (height, width, 3) uint8 numpy array containing RGB values (or None).
        * A (height, width) float32 numpy array containing depth values (or None).
    """

    # Update scene geometry.
    self.update(scene_option=scene_option)

    # Render scene and text overlays, read contents of RGB or depth buffer.
    #self._simulator.contexts.gl.make_current()
    self._render_on_gl_thread()

    depth_image = None
    rgb_image = None
    if self._depth:
      # Get the distances to the near and far clipping planes.
      #extent = self._physics.model.stat.extent
      #near = self._physics.model.vis.map.znear * extent
      #far = self._physics.model.vis.map.zfar * extent
      # Convert from [0 1] to depth in meters, see links below:
      # http://stackoverflow.com/a/6657284/1461210
      # https://www.khronos.org/opengl/wiki/Depth_Buffer_Precision
      #image = near / (1 - self._depth_buffer * (1 - near / far))
      depth_image = np.flipud(self._depth_buffer).copy()
    if self._rgb:
      rgb_image = np.flipud(self._rgb_buffer).copy()

    # The first row in the buffer is the bottom row of pixels in the image.
    return rgb_image, depth_image

  def set_fps(self, dt):
    """Sets the frames per second value, required for recording videos.
    Args:
      dt: Time elapsed between two rendered images
    """
    self._fps = int(np.round(1.0 / dt))

  def write_video(self, imgs, filepath):
    # DEPRECATED -> use write_video_set_path(filepath), write_video_add_frame(img), and write_video_close() instead
    """Writes a video from images.
    Args:
      imgs: A list of images.
      filepath: Path where the video will be saved.
    Raises:
      ValueError: If frames per second (fps) is not set (set_fps is not called)
    """

    # Make sure fps has been set
    if self._fps is None:
      raise ValueError("set_fps must be called before writing a video.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, self._fps, tuple(self._resolution))
    for img in imgs:
      out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out.release()

  def write_video_set_path(self, filepath):
    """Creates a video writer instance that can be used online to add frames
    (might be less memory consuming than write_video()).
    Args:
      filepath: Path where the video will be saved.
    Raises:
      ValueError: If frames per second (fps) is not set (set_fps is not called)
    """

    # Make sure fps has been set
    if self._fps is None:
      raise ValueError("set_fps must be called before writing a video.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    self._write_video_out = cv2.VideoWriter(filepath, fourcc, self._fps, tuple(self._resolution))

  def write_video_add_frame(self, img):
    """Adds the given frame to the video writer instance
    (might be less memory consuming than write_video()).
    Args:
      img: The image to write.
    """
    assert hasattr(self, "_write_video_out"), "Call 'write_video_set_path(filepath)' first to create a video writer instance."
    self._write_video_out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

  def write_video_close(self):
    """Closes the existing video writer instance
    (might be less memory consuming than write_video()).
    """
    assert hasattr(self, "_write_video_out"), "Call 'write_video_set_path(filepath)' first to create a video writer instance."
    self._write_video_out.release()