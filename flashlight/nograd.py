from .Config import Config

class no_grad:
  def __enter__(self):
    Config.enable_backprop = False

  def __exit__(self, exc_type, exc_value, traceback):
    Config.enable_backprop = True