from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.layers import base as base_layer
from tensorflow.python.platform import flags

flags.DEFINE_boolean("usefft", True, "set whether to use fft calculation")
FLAGS = flags.FLAGS
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


def _block_circulant_dot(x, w, output_dim):
  print("block_circulant_begin\n------------------------------------")
  partion_input = w.shape.as_list()[0]
  partion_output = w.shape.as_list()[1]
  partition_size = w.shape.as_list()[2]
  input_dim = x.shape.as_list()[1]
  # print(partion_input)
  # print(partion_output)
  # print(partition_size)
  # print(batch_size)
  # print(timesteps)
  # print(input_dim)
  x = tf.pad(x, [[0, 0], [0, partition_size * partion_input - input_dim]],
             "CONSTANT")
  x = tf.reshape(x, [-1, partion_input, 1, partition_size])
  print("w:", w.shape.as_list())
  print("x:", x.shape.as_list())
  # complex_w = tf.complex(w,tf.zeros_like(w))
  # complex_x = tf.complex(x,tf.zeros_like(x))
  # print('complex_w:', complex_w.shape.as_list())
  # print('complex_x:', complex_x.shape.as_list())
  # ret = tf.multiply(tf.fft(complex_w),tf.fft(complex_x))
  ret = tf.multiply(tf.spectral.rfft(w), tf.spectral.rfft(x))
  print(ret.shape.as_list())
  ret = tf.spectral.irfft(ret)
  # ret = tf.ifft(ret)
  # ret = tf.real(ret)
  ret = tf.reduce_sum(ret, axis=1)
  print(ret.shape.as_list())
  ret = tf.reshape(ret, [-1, partition_size * partion_output])
  print(ret.shape.as_list())
  ret = ret[:, :output_dim]
  print(ret.shape.as_list())
  print("block_circulant_end\n----------------------------------")
  return ret


def calculate_indx(input_dim, output_dim, p_size):
  num_filters_out = output_dim
  num_filters_in = input_dim

  def block_indx(k, rc, cc):
    rc = int((rc + k - 1) // k) * k
    cc = int((cc + k - 1) // k) * k
    i = np.arange(0, k, 1).reshape([1, k])
    j = np.arange(0, -k, -1).reshape([k, 1])
    indx = i + j
    indx = (indx + k) % k
    m = np.tile(indx, [int(rc // k), int(cc // k)])
    offset = np.arange(0, rc * cc)
    i = (offset // cc) // k
    j = (offset % cc) // k
    offset = (i * cc + j * k).reshape([rc, cc])
    return m + offset

  if p_size and p_size <= np.min([num_filters_out, num_filters_in]):
    indx = block_indx(p_size, num_filters_in, num_filters_out)
    target_c = num_filters_in * num_filters_out // p_size
    print("you are using BlockCirc", p_size)
  else:
    print("sorry, not enough size for partitoning", num_filters_in,
          num_filters_out)
    target_c = np.max([num_filters_in, num_filters_out])
    a, b = np.ogrid[0:target_c, 0:-target_c:-1]
    indx = a + b
  print("num_filters_in:{}".format(num_filters_in))
  print("num_filters_out:{}".format(num_filters_out))
  print("target_c:{}".format(target_c))
  indx = (indx + target_c) % target_c
  print(indx)
  indx = tf.constant(indx[:num_filters_in, :num_filters_out].astype(np.int32))
  return target_c, indx


def _add_variable(input_dim,
                  output_dim,
                  name,
                  add_fn,
                  initializer,
                  partitioner,
                  partition_size=None):
  if partition_size:
    if FLAGS.usefft:
      partition_row = (input_dim + partition_size - 1) // partition_size
      partition_col = (output_dim + partition_size - 1) // partition_size
      weights = add_fn(
          name,
          shape=[partition_row, partition_col, partition_size],
          partitioner=partitioner,
          initializer=initializer)
    else:
      weight_shape, kernel_indx = calculate_indx(input_dim, output_dim,
                                                 partition_size)
      weights = add_fn(
          name,
          shape=[weight_shape],
          partitioner=partitioner,
          initializer=initializer)
      weights = tf.gather(weights, kernel_indx)
      weights = tf.reshape(weights, [input_dim, output_dim])
  else:
    weights = add_fn(
        name,
        shape=[input_dim, output_dim],
        partitioner=partitioner,
        initializer=initializer)
  return weights


def matrix_mul(inputs, weights, output_size=None, partition_size=None):
  if partition_size and FLAGS.usefft:
    matrix = _block_circulant_dot(inputs, weights, output_size)
  else:
    matrix = tf.matmul(inputs, weights)
  return matrix


class BlockCirculantGRUCell(tf.nn.rnn_cell.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
    dtype: Default dtype of the layer (default of `None` means use the type
      of the first input). Required when `build` is called before `call`.
  """

  def __init__(self,
               num_units,
               partition_size=None,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None,
               dtype=None):
    super(BlockCirculantGRUCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)
    self.partition_size = partition_size
    self._num_units = num_units
    self._activation = activation or tf.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError(
          "Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

    input_depth = inputs_shape[1].value
    self._gate_kernel = _add_variable(
        input_dim=input_depth + self._num_units,
        output_dim=2 * self._num_units,
        name="gates/%s" % _WEIGHTS_VARIABLE_NAME,
        add_fn=self.add_variable,
        initializer=self._kernel_initializer,
        partitioner=None,
        partition_size=self.partition_size)
    self._gate_bias = self.add_variable(
        "gates/%s" % _BIAS_VARIABLE_NAME,
        shape=[2 * self._num_units],
        initializer=(self._bias_initializer
                     if self._bias_initializer is not None else
                     tf.constant_initializer(1.0, dtype=self.dtype)))
    self._candidate_kernel = _add_variable(
        input_dim=input_depth + self._num_units,
        output_dim=self._num_units,
        name="candidate/%s" % _WEIGHTS_VARIABLE_NAME,
        add_fn=self.add_variable,
        initializer=self._kernel_initializer,
        partitioner=None,
        partition_size=self.partition_size)
    self._candidate_bias = self.add_variable(
        "candidate/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=(self._bias_initializer
                     if self._bias_initializer is not None else
                     tf.zeros_initializer(dtype=self.dtype)))

    self.built = True

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""

    gate_inputs = matrix_mul(
        tf.concat([inputs, state], 1), self._gate_kernel, 2 * self._num_units,
        self.partition_size)
    gate_inputs = tf.nn.bias_add(gate_inputs, self._gate_bias)

    value = tf.sigmoid(gate_inputs)
    r, u = tf.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state

    candidate = matrix_mul(
        tf.concat([inputs, r_state], 1), self._candidate_kernel,
        self._num_units, self.partition_size)

    candidate = tf.nn.bias_add(candidate, self._candidate_bias)

    c = self._activation(candidate)
    new_h = u * state + (1 - u) * c
    return new_h, new_h


class BlockCirculantLSTMCell(tf.nn.rnn_cell.RNNCell):
  """Long short-term memory unit (LSTM) recurrent network cell.
  The default non-peephole implementation is based on:
    http://www.bioinf.jku.at/publications/older/2604.pdf
  S. Hochreiter and J. Schmidhuber.
  "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
  The peephole implementation is based on:
    https://research.google.com/pubs/archive/43905.pdf
  Hasim Sak, Andrew Senior, and Francoise Beaufays.
  "Long short-term memory recurrent neural network architectures for
   large scale acoustic modeling." INTERSPEECH, 2014.
  The class uses optional peep-hole connections, optional cell clipping, and
  an optional projection layer.
  """

  def __init__(self,
               num_units,
               partition_size=None,
               use_peepholes=False,
               cell_clip=None,
               initializer=None,
               num_proj=None,
               proj_clip=None,
               num_unit_shards=None,
               num_proj_shards=None,
               forget_bias=1.0,
               state_is_tuple=True,
               activation=None,
               reuse=None,
               name=None,
               dtype=None):
    """Initialize the parameters for an LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
        provided, then the projected values are clipped elementwise to within
        `[-proj_clip, proj_clip]`.
      num_unit_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      num_proj_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training. Must set it manually to `0.0` when restoring from
        CudnnLSTM trained checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  This latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
      dtype: Default dtype of the layer (default of `None` means use the type
        of the first input). Required when `build` is called before `call`.
      When restoring from CudnnLSTM-trained checkpoints, use
      `CudnnCompatibleLSTMCell` instead.
    """
    super(BlockCirculantLSTMCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype)
    if not state_is_tuple:
      tf.logging.warn(
          "%s: Using a concatenated state is slower and will soon be "
          "deprecated.  Use state_is_tuple=True.", self)
    if num_unit_shards is not None or num_proj_shards is not None:
      tf.logging.warn(
          "%s: The num_unit_shards and proj_unit_shards parameters are "
          "deprecated and will be removed in Jan 2017.  "
          "Use a variable scope with a partitioner instead.", self)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)
    self.partition_size = partition_size
    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or tf.tanh

    if num_proj:
      self._state_size = (
          tf.contrib.rnn.LSTMStateTuple(num_units, num_proj)
          if state_is_tuple else num_units + num_proj)
      self._output_size = num_proj
    else:
      self._state_size = (
          tf.contrib.rnn.LSTMStateTuple(num_units, num_units)
          if state_is_tuple else 2 * num_units)
      self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError(
          "Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

    input_depth = inputs_shape[1].value
    h_depth = self._num_units if self._num_proj is None else self._num_proj
    maybe_partitioner = (
        tf.fixed_size_partitioner(self._num_unit_shards)
        if self._num_unit_shards is not None else None)
    self._kernel = _add_variable(
        input_dim=input_depth + h_depth,
        output_dim=4 * self._num_units,
        name=_WEIGHTS_VARIABLE_NAME,
        add_fn=self.add_variable,
        initializer=self._initializer,
        partitioner=maybe_partitioner,
        partition_size=self.partition_size)
    if self.dtype is None:
      initializer = tf.zeros_initializer
    else:
      initializer = tf.zeros_initializer(dtype=self.dtype)
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[4 * self._num_units],
        initializer=initializer)
    if self._use_peepholes:
      self._w_f_diag = self.add_variable(
          "w_f_diag", shape=[self._num_units], initializer=self._initializer)
      self._w_i_diag = self.add_variable(
          "w_i_diag", shape=[self._num_units], initializer=self._initializer)
      self._w_o_diag = self.add_variable(
          "w_o_diag", shape=[self._num_units], initializer=self._initializer)

    if self._num_proj is not None:
      maybe_proj_partitioner = (
          tf.fixed_size_partitioner(self._num_proj_shards)
          if self._num_proj_shards is not None else None)
      self._proj_kernel = _add_variable(
          input_dim=self._num_units,
          output_dim=self._num_proj,
          name="projection/{}".format(_WEIGHTS_VARIABLE_NAME),
          add_fn=self.add_variable,
          initializer=self._initializer,
          partitioner=maybe_proj_partitioner,
          partition_size=self.partition_size)

    self.built = True

  def call(self, inputs, state):
    """Run one step of LSTM.
    Args:
      inputs: input Tensor, 2D, `[batch, num_units].
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, [batch, state_size]`.  If `state_is_tuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `m_state`.
    Returns:
      A tuple containing:
      - A `2-D, [batch, output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.
    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    num_proj = self._num_units if self._num_proj is None else self._num_proj
    sigmoid = tf.sigmoid

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = tf.slice(state, [0, 0], [-1, self._num_units])
      m_prev = tf.slice(state, [0, self._num_units], [-1, num_proj])

    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate

    lstm_matrix = matrix_mul(
        tf.concat([inputs, m_prev], 1), self._kernel, 4 * self._num_units,
        self.partition_size)

    lstm_matrix = tf.nn.bias_add(lstm_matrix, self._bias)

    i, j, f, o = tf.split(value=lstm_matrix, num_or_size_splits=4, axis=1)
    # Diagonal connections
    if self._use_peepholes:
      c = (
          sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
          sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
    else:
      c = (
          sigmoid(f + self._forget_bias) * c_prev +
          sigmoid(i) * self._activation(j))

    if self._cell_clip is not None:
      # pylint: disable=invalid-unary-operand-type
      c = tf.clip_by_value(c, -self._cell_clip, self._cell_clip)
      # pylint: enable=invalid-unary-operand-type
    if self._use_peepholes:
      m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
    else:
      m = sigmoid(o) * self._activation(c)

    if self._num_proj is not None:
      m = matrix_mul(m, self._proj_kernel, self._num_proj, self.partition_size)
      if self._proj_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        m = tf.clip_by_value(m, -self._proj_clip, self._proj_clip)
        # pylint: enable=invalid-unary-operand-type

    new_state = (
        tf.contrib.rnn.LSTMStateTuple(c, m)
        if self._state_is_tuple else tf.concat([c, m], 1))
    return m, new_state
