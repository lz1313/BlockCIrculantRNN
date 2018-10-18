import time
import datetime
import os
from functools import wraps

import numpy as np
import tensorflow.google as tf

from model import DRNN
from utils import output_to_sequence
from utils import get_edit_distance
from utils import load_batched_data
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
flags.DEFINE_string('master', 'local',
                    """BNS name of the TensorFlow runtime to use.""")
flags.DEFINE_bool('is_training', True, 'set whether to train or test')
flags.DEFINE_boolean(
    'restore', False,
    'set whether to restore a model, when test mode, keep should be set to True'
)
flags.DEFINE_string('level', 'phn', 'set the task level, phn, cha.')

flags.DEFINE_string('cell', 'LSTM', 'set the rnncell to use, GRU, LSTM...')
flags.DEFINE_string('activation', 'tanh',
                    'set the activation to use, sigmoid, tanh, relu, elu...')

flags.DEFINE_integer('batch_size', 32, 'set the batch size')
flags.DEFINE_integer('num_hidden', 1024, 'set the hidden size of rnn cell')
flags.DEFINE_bool('use_peepholes', True, 'set whether to use peephole')
flags.DEFINE_integer('feature_length', 39, 'set the size of input feature')
flags.DEFINE_integer('num_classes', 62, 'set the number of output classes')
flags.DEFINE_integer('num_proj', 512, 'set the number of output classes')
flags.DEFINE_integer('num_epochs', 500, 'set the number of epochs')
flags.DEFINE_float('learning_rate', 1e-5, 'set the learning rate')
flags.DEFINE_float('keep_prob', 0.7, 'set probability of keep in dropout')
flags.DEFINE_float(
    'clip_gradient_norm', 1,
    'set the threshold of gradient clipping, -1 denotes no clipping')
flags.DEFINE_integer('partition_size', None, 'set partition_size for rnn')
flags.DEFINE_string('input_data_dir',
                    'timit_preproc',
                    'set the data root directory')
flags.DEFINE_string('exp_dir', 'timit_exp',
                    'set the log directory')

FLAGS = flags.FLAGS


def log_scalar(writer, tag, value, step):
  """Log a scalar variable.
  Parameter
  ----------
  tag : basestring
      Name of the scalar
  value
  step : int
      training iteration
  """
  summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
  writer.add_summary(summary, step)


def main(_):
  train_mfcc_dir = os.path.join(FLAGS.input_data_dir, FLAGS.level, 'TRAIN',
                                'mfcc')
  train_label_dir = os.path.join(FLAGS.input_data_dir, FLAGS.level, 'TRAIN',
                                 'label')
  test_mfcc_dir = os.path.join(FLAGS.input_data_dir, FLAGS.level, 'TEST',
                               'mfcc')
  test_label_dir = os.path.join(FLAGS.input_data_dir, FLAGS.level, 'TEST',
                                'label')

  savedir = os.path.join(FLAGS.exp_dir, FLAGS.level, 'save')
  resultdir = os.path.join(FLAGS.exp_dir, FLAGS.level, 'result')

  if FLAGS.is_training:
    batched_data, max_time_steps, total_n = load_batched_data(
        train_mfcc_dir, train_label_dir, FLAGS.batch_size, FLAGS.level)
  else:
    batched_data, max_time_steps, total_n = load_batched_data(
        test_mfcc_dir, test_label_dir, FLAGS.batch_size, FLAGS.level)

  hparams = {}
  hparams['level'] = FLAGS.level
  hparams['batch_size'] = FLAGS.batch_size
  hparams['partition_size'] = FLAGS.partition_size
  hparams['num_hidden'] = FLAGS.num_hidden
  hparams['feature_length'] = FLAGS.feature_length
  hparams['num_classes'] = FLAGS.num_classes
  hparams['num_proj'] = FLAGS.num_proj
  hparams['learning_rate'] = FLAGS.learning_rate
  hparams['keep_prob'] = FLAGS.keep_prob
  hparams['clip_gradient_norm'] = FLAGS.clip_gradient_norm
  hparams['use_peepholes'] = FLAGS.use_peepholes
  if FLAGS.activation == 'tanh':
    hparams['activation'] = tf.tanh
  elif FLAGS.activation == 'relu':
    hparams['activation'] = tf.nn.relu
  hparams['max_time_steps'] = max_time_steps
  with tf.Graph().as_default():
    model = DRNN(FLAGS.cell, hparams, FLAGS.is_training)
    train_writer = tf.summary.FileWriter(resultdir + '/train')
    test_writer = tf.summary.FileWriter(resultdir + '/test')
    with tf.Session(FLAGS.master) as sess:
      # restore from stored model
      if FLAGS.restore:
        ckpt = tf.train.get_checkpoint_state(savedir)
        if ckpt and ckpt.model_checkpoint_path:
          model.saver.restore(sess, ckpt.model_checkpoint_path)
          print('Model restored from:' + ckpt.model_checkpoint_path)
      else:
        print('Initializing')
        sess.run(model.initial_op)
      train_writer.add_graph(sess.graph)
      for epoch in range(FLAGS.num_epochs):
        ## training
        start = time.time()
        if FLAGS.is_training:
          print('Epoch', epoch + 1, '...')
        batch_errors = np.zeros(len(batched_data))
        batched_random_idx = np.random.permutation(len(batched_data))

        for batch, batch_original_idx in enumerate(batched_random_idx):
          batch_inputs, batch_target_sparse, batch_seq_length = batched_data[
              batch_original_idx]
          batch_tgt_idx, batch_tgt_vals, batch_tgt_shape = batch_target_sparse
          feeddict = {
              model.x: batch_inputs,
              model.tgt_idx: batch_tgt_idx,
              model.tgt_vals: batch_tgt_vals,
              model.tgt_shape: batch_tgt_shape,
              model.seq_length: batch_seq_length
          }

          if FLAGS.is_training and (
              (epoch * len(batched_random_idx) + batch + 1) % 20 == 0 or
              (epoch == FLAGS.num_epochs - 1 and
               batch == len(batched_random_idx) - 1)):
            checkpoint_path = os.path.join(savedir, 'model.ckpt')
            model.saver.save(
                sess, checkpoint_path, global_step=model.global_step)
            print('Model has been saved in {}'.format(savedir))

          if FLAGS.level == 'cha':
            if FLAGS.is_training:
              _, l, pre, y, er, global_step = sess.run(
                  [
                      model.train_op, model.loss, model.predictions, model.y,
                      model.error_rate, model.global_step
                  ],
                  feed_dict=feeddict)
              batch_errors[batch] = er
              if global_step % 10 == 0:
                log_scalar(train_writer, 'CER', er / FLAGS.batch_size,
                           global_step)
                print('{} mode, global_step:{}, lr:{}, total:{}, '
                      'batch:{}/{},epoch:{}/{},train loss={:.3f},mean train '
                      'CER={:.3f}'.format(
                          FLAGS.level, global_step,
                          FLAGS.learning_rate, total_n, batch + 1,
                          len(batched_random_idx), epoch + 1, FLAGS.num_epochs,
                          l, er / FLAGS.batch_size))

            elif not FLAGS.is_training:
              l, pre, y, er, global_step = sess.run(
                  [
                      model.loss, model.predictions, model.y, model.error_rate,
                      model.global_step
                  ],
                  feed_dict=feeddict)
              batch_errors[batch] = er
              log_scalar(test_writer, 'CER', er / FLAGS.batch_size, global_step)
              print('{} mode, global_step:{}, total:{}, batch:{}/{},test '
                    'loss={:.3f},mean test CER={:.3f}'.format(
                        FLAGS.level, global_step, total_n, batch + 1,
                        len(batched_random_idx), l, er / FLAGS.batch_size))

          elif FLAGS.level == 'phn':
            if FLAGS.is_training:
              _, l, pre, y, global_step = sess.run(
                  [
                      model.train_op, model.loss, model.predictions, model.y,
                      model.global_step
                  ],
                  feed_dict=feeddict)
              er = get_edit_distance([pre.values], [y.values], True,
                                     FLAGS.level)
              if global_step % 10 == 0:
                log_scalar(train_writer, 'PER', er, global_step)
                print(
                    '{} mode, global_step:{}, lr:{}, total:{}, '
                    'batch:{}/{},epoch:{}/{},train loss={:.3f},mean train '
                    'PER={:.3f}'.format(FLAGS.level, global_step,
                                        FLAGS.learning_rate, total_n, batch + 1,
                                        len(batched_random_idx), epoch + 1,
                                        FLAGS.num_epochs, l, er))
              batch_errors[batch] = er * len(batch_seq_length)
            elif not FLAGS.is_training:
              l, pre, y, global_step = sess.run(
                  [model.loss, model.predictions, model.y, model.global_step],
                  feed_dict=feeddict)
              er = get_edit_distance([pre.values], [y.values], True,
                                     FLAGS.level)
              log_scalar(test_writer, 'PER', er, global_step)
              print('{} mode, global_step:{}, total:{}, batch:{}/{},test '
                    'loss={:.3f},mean test PER={:.3f}'.format(
                        FLAGS.level, global_step, total_n, batch + 1,
                        len(batched_random_idx), l, er))
              batch_errors[batch] = er * len(batch_seq_length)

          # NOTE:
          if er / FLAGS.batch_size == 1.0:
            break

          if batch % 100 == 0:
            print('Truth:\n' + output_to_sequence(y, level=FLAGS.level))
            print('Output:\n' + output_to_sequence(pre, level=FLAGS.level))

        end = time.time()
        delta_time = end - start
        print('Epoch ' + str(epoch + 1) + ' needs time:' + str(delta_time) +
              ' s')

        if FLAGS.is_training:
          if (epoch + 1) % 1 == 0:
            checkpoint_path = os.path.join(savedir, 'model.ckpt')
            model.saver.save(
                sess, checkpoint_path, global_step=model.global_step)
            print('Model has been saved in {}'.format(savedir))
          epoch_er = batch_errors.sum() / total_n
          print('Epoch', epoch + 1, 'mean train error rate:', epoch_er)

        if not FLAGS.is_training:
          with tf.gfile.GFile(
              os.path.join(resultdir, FLAGS.level + '_result.txt'),
              'a') as result:
            result.write(output_to_sequence(y, level=FLAGS.level) + '\n')
            result.write(output_to_sequence(pre, level=FLAGS.level) + '\n')
            result.write('\n')
          epoch_er = batch_errors.sum() / total_n
          print(' test error rate:', epoch_er)


if __name__ == '__main__':
  app.run()
