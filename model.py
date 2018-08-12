import tensorflow as tf
from block_circulant_cell import BlockCirculantLSTMCell
from block_circulant_cell import BlockCirculantGRUCell
slim = tf.contrib.slim


class DRNN(object):

  def __init__(self, cell_fn, hparams, is_training=True):
    self.is_training = is_training
    self.loss = None
    self.metrics = None
    self.hparams = hparams
    self.cell_fn = cell_fn
    self.build_model()
    self.build_loss()
    self.build_eval_metrics()
    self.saver = tf.train.Saver()
    self.global_step = tf.train.get_or_create_global_step()
    self.initial_op = tf.global_variables_initializer()

  def build_model(self):
    self.x = tf.placeholder(
        tf.float32,
        shape=(self.hparams['max_time_steps'], self.hparams['batch_size'],
               self.hparams['feature_length']))
    self.input_list = tf.split(
        tf.reshape(self.x, [-1, self.hparams['feature_length']]),
        self.hparams['max_time_steps'],
        axis=0)
    self.tgt_idx = tf.placeholder(tf.int64)
    self.tgt_vals = tf.placeholder(tf.int32)
    self.tgt_shape = tf.placeholder(tf.int64)
    self.y = tf.SparseTensor(self.tgt_idx, self.tgt_vals, self.tgt_shape)
    self.seq_length = tf.placeholder(tf.int32, self.hparams['batch_size'])

    # build RNN
    # RNN #1
    if self.cell_fn == 'LSTM':
      forward_cell_1 = BlockCirculantLSTMCell(
          self.hparams['num_hidden'],
          partition_size=self.hparams['partition_size'],
          use_peepholes=self.hparams['use_peepholes'],
          num_proj=self.hparams['num_proj'],
          activation=self.hparams['activation'])
      backward_cell_1 = BlockCirculantLSTMCell(
          self.hparams['num_hidden'],
          partition_size=self.hparams['partition_size'],
          use_peepholes=self.hparams['use_peepholes'],
          num_proj=self.hparams['num_proj'],
          activation=self.hparams['activation'])
    elif self.cell_fn == 'GRU':
      forward_cell_1 = BlockCirculantGRUCell(
          self.hparams['num_hidden'],
          partition_size=self.hparams['partition_size'],  
          activation=self.hparams['activation'])
      backward_cell_1 = BlockCirculantGRUCell(
          self.hparams['num_hidden'],
          partition_size=self.hparams['partition_size'],  
          activation=self.hparams['activation'])
    else:
      raise ValueError('cell not supported')

    outputs_1, _ = tf.nn.bidirectional_dynamic_rnn(
        forward_cell_1,
        backward_cell_1,
        inputs=self.x,
        dtype=tf.float32,
        sequence_length=self.seq_length,
        time_major=True,
        scope='RNN_1')
    output_fw_1, output_bw_1 = outputs_1

    output_fb_1 = tf.concat([output_fw_1, output_bw_1], 2)
    shape = output_fb_1.get_shape().as_list()
    output_fb_1 = tf.reshape(
        output_fb_1,
        [shape[0], shape[1], 2, int(shape[2] / 2)])
    hidden_1 = tf.reduce_sum(output_fb_1, 2)
    hidden_1 = tf.layers.dropout(
        hidden_1, rate=self.hparams['keep_prob'], training=self.is_training)
    # RNN #2
    if self.cell_fn == 'LSTM':
      forward_cell_2 = BlockCirculantLSTMCell(
          self.hparams['num_hidden'],
          partition_size=self.hparams['partition_size'],
          use_peepholes=self.hparams['use_peepholes'],
          num_proj=self.hparams['num_proj'],
          activation=self.hparams['activation'])
      backward_cell_2 = BlockCirculantLSTMCell(
          self.hparams['num_hidden'],
          partition_size=self.hparams['partition_size'],
          use_peepholes=self.hparams['use_peepholes'],
          num_proj=self.hparams['num_proj'],
          activation=self.hparams['activation'])
    elif self.cell_fn == 'GRU':
      forward_cell_2 = BlockCirculantGRUCell(
          self.hparams['num_hidden'],
          partition_size=self.hparams['partition_size'],
          activation=self.hparams['activation'])
      backward_cell_2 = BlockCirculantGRUCell(
          self.hparams['num_hidden'],
          partition_size=self.hparams['partition_size'],
          activation=self.hparams['activation'])
    else:
      raise ValueError('cell not supported')

    outputs_2, _ = tf.nn.bidirectional_dynamic_rnn(
        forward_cell_2,
        backward_cell_2,
        inputs=hidden_1,
        dtype=tf.float32,
        sequence_length=self.seq_length,
        time_major=True,
        scope='RNN_2')
    output_fw_2, output_bw_2 = outputs_2

    output_fb_2 = tf.concat([output_fw_2, output_bw_2], 2)
    shape = output_fb_2.get_shape().as_list()
    output_fb_2 = tf.reshape(
        output_fb_2,
        [shape[0], shape[1], 2, int(shape[2] / 2)])
    hidden_2 = tf.reduce_sum(output_fb_2, 2)
    hidden_2 = tf.layers.dropout(
        hidden_2, rate=self.hparams['keep_prob'], training=self.is_training)

    outputs = tf.reshape(hidden_2, [-1, self.hparams['num_hidden']])
    output_list = tf.split(outputs, self.hparams['max_time_steps'], 0)
    output_probs = [
        tf.reshape(t, [
            self.hparams['batch_size'], self.hparams['num_proj']
            if self.hparams['num_proj'] else self.hparams['num_hidden']
        ]) for t in output_list
    ]
    with tf.name_scope('fc-layer'):
      with tf.variable_scope('fc'):
        logits_w = tf.Variable(
            tf.truncated_normal(
                [
                    self.hparams['num_proj'] if self.hparams['num_proj'] else
                    self.hparams['num_hidden'], self.hparams['num_classes']
                ],
                name='logits/weights'))
        logits_b = tf.Variable(
            tf.zeros([self.hparams['num_classes']]), name='logits/biases')
        logits = [tf.matmul(t, logits_w) + logits_b for t in output_probs]
    self.logits3d = tf.stack(logits)

  def build_loss(self):
    self.loss = tf.reduce_mean(
        tf.nn.ctc_loss(self.y, self.logits3d, self.seq_length))
    opt = tf.train.AdamOptimizer(self.hparams['learning_rate'])
    clip_gradient_norm = self.hparams[
        'clip_gradient_norm'] if self.hparams[
        'clip_gradient_norm'] > 0 else 0
    self.train_op = slim.learning.create_train_op(
        self.loss, opt, clip_gradient_norm=clip_gradient_norm)

  def build_eval_metrics(self):
    self.predictions = tf.to_int32(
        tf.nn.ctc_beam_search_decoder(
            self.logits3d, self.seq_length, merge_repeated=False)[0][0])
    self.error_rate = tf.reduce_sum(
        tf.edit_distance(self.predictions, self.y, normalize=True))
