class Config:
  enable_backprop = True
  linear_layer_count = 0
  batch_norm_1d_count = 0
  tanh_count = 0
  neural_net_count = 0

  @staticmethod
  def increment_linear_layer_count():
    Config.linear_layer_count += 1
    return Config.linear_layer_count

  @staticmethod
  def increment_batch_norm_1d_count():
    Config.batch_norm_1d_count += 1
    return Config.batch_norm_1d_count

  @staticmethod
  def increment_tanh_count():
    Config.tanh_count += 1
    return Config.tanh_count

  @staticmethod
  def increment_neural_net_count():
    Config.neural_net_count += 1
    return Config.neural_net_count