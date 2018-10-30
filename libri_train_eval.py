import time
import datetime
import os
from six.moves import cPickle
from functools import wraps
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc

from utils import load_batched_data, output_to_sequence, count_params, check_path_exists
from model import DRNN


from tensorflow.python.platform import flags
from tensorflow.python.platform import app

flags.DEFINE_string('train_dataset', 'train-clean-100', 'set the training dataset')
flags.DEFINE_string('dev_dataset', 'dev-clean', 'set the development dataset')
flags.DEFINE_string('test_dataset', 'test-clean', 'set the test dataset')

flags.DEFINE_string('master', '',
                    """BNS name of the TensorFlow runtime to use.""")
flags.DEFINE_string('mode', 'train', 'set whether to train, dev or test')
flags.DEFINE_boolean(
    'restore', False,
    'set whether to restore a model, when test mode, keep should be set to True'
)
flags.DEFINE_string('level', 'cha', 'set the task level, phn, cha.')

flags.DEFINE_string('cell', 'LSTM', 'set the rnncell to use, GRU, LSTM...')
flags.DEFINE_string('activation', 'tanh',
                    'set the activation to use, sigmoid, tanh, relu, elu...')

flags.DEFINE_integer('batch_size', 16, 'set the batch size')
flags.DEFINE_integer('num_hidden', 256, 'set the hidden size of rnn cell')
flags.DEFINE_bool('use_peepholes', True, 'set whether to use peephole')
flags.DEFINE_integer('feature_length', 39, 'set the size of input feature')
flags.DEFINE_integer('num_classes', 29, 'set the number of output classes')
flags.DEFINE_integer('num_proj', 128, 'set the number of output classes')
flags.DEFINE_integer('num_epochs', 500, 'set the number of epochs')
flags.DEFINE_float('learning_rate', 1e-4, 'set the learning rate')
flags.DEFINE_float('keep_prob', 0.7, 'set probability of keep in dropout')
flags.DEFINE_float(
    'clip_gradient_norm', 1,
    'set the threshold of gradient clipping, -1 denotes no clipping')
flags.DEFINE_integer('partition_size', 4, 'set partition_size for rnn')
flags.DEFINE_string('input_data_dir',
                    'libri_process',
                    'set the data root directory')
flags.DEFINE_string('exp_dir', 'libri_exp_block_4',
                    'set the log directory')

FLAGS = flags.FLAGS


def get_data(datadir, level, train_dataset, dev_dataset, test_dataset, mode):
    if mode == 'train':
        train_feature_dirs = [os.path.join(os.path.join(datadir, level, train_dataset), 
            i, 'feature') for i in os.listdir(os.path.join(datadir, level, train_dataset))]

        train_label_dirs = [os.path.join(os.path.join(datadir, level, train_dataset), 
            i, 'label') for i in os.listdir(os.path.join(datadir, level, train_dataset))]
        return train_feature_dirs, train_label_dirs

    if mode == 'dev':
        dev_feature_dirs = [os.path.join(os.path.join(datadir, level, dev_dataset), 
            i, 'feature') for i in os.listdir(os.path.join(datadir, level, dev_dataset))]

        dev_label_dirs = [os.path.join(os.path.join(datadir, level, dev_dataset), 
            i, 'label') for i in os.listdir(os.path.join(datadir, level, dev_dataset))]
        return dev_feature_dirs, dev_label_dirs

    if mode == 'test':
        test_feature_dirs = [os.path.join(os.path.join(datadir, level, test_dataset), 
            i, 'feature') for i in os.listdir(os.path.join(datadir, level, test_dataset))]

        test_label_dirs = [os.path.join(os.path.join(datadir, level, test_dataset), 
            i, 'label') for i in os.listdir(os.path.join(datadir, level, test_dataset))]
        return test_feature_dirs, test_label_dirs

def main(_):
    print('%s mode...'%str(FLAGS.mode))
    savedir = os.path.join(FLAGS.exp_dir, FLAGS.level, 'save')
    resultdir = os.path.join(FLAGS.exp_dir, FLAGS.level, 'result')
    check_path_exists([savedir, resultdir])
    # load data
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
    feature_dirs, label_dirs = get_data(FLAGS.input_data_dir, FLAGS.level, FLAGS.train_dataset, FLAGS.dev_dataset, FLAGS.test_dataset, FLAGS.mode)
    batched_data, max_time_steps, total_n = load_batched_data(feature_dirs[0], label_dirs[0], FLAGS.batch_size, FLAGS.level)
    hparams['max_time_steps'] = max_time_steps
    ## shuffle feature_dir and label_dir by same order
    FL_pair = list(zip(feature_dirs, label_dirs))
    random.shuffle(FL_pair)
    feature_dirs, label_dirs = zip(*FL_pair)
    train_writer = tf.summary.FileWriter(resultdir + '/train')
    test_writer = tf.summary.FileWriter(resultdir + '/test')

    for feature_dir, label_dir in zip(feature_dirs, label_dirs):
        id_dir = feature_dirs.index(feature_dir)
        print('dir id:{}'.format(id_dir))
        batched_data, max_time_steps, total_n = load_batched_data(feature_dir, label_dir, FLAGS.batch_size, FLAGS.level)
        hparams['max_time_steps'] = max_time_steps
        model = DRNN(FLAGS.cell, hparams, FLAGS.mode == 'train')

        with tf.Session(FLAGS.master) as sess:
            # restore from stored model
            if FLAGS.restore:
                ckpt = tf.train.get_checkpoint_state(savedir)
                if ckpt and ckpt.model_checkpoint_path:
                    model.saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Model restored from:' + savedir)
            else:
                print('Initializing')
                sess.run(model.initial_op)

            for epoch in range(FLAGS.num_epochs):
                ## training
                start = time.time()
                if FLAGS.mode == 'train':
                    print('Epoch {} ...'.format(epoch + 1))

                batch_errors = np.zeros(len(batched_data))
                batched_random_idx = np.random.permutation(len(batched_data))

                for batch, batch_original_idx in enumerate(batched_random_idx):
                    batch_inputs, batch_target_sparse, batch_seq_length = batched_data[batch_original_idx]
                    batch_tgt_idx, batch_tgt_vals, batch_tgt_shape = batch_target_sparse
                    feedDict = {
                      model.x: batch_inputs,
                      model.tgt_idx: batch_tgt_idx,
                      model.tgt_vals: batch_tgt_vals,
                      model.tgt_shape: batch_tgt_shape,
                      model.seq_length: batch_seq_length
                    }

                    if FLAGS.level == 'cha':
                        if FLAGS.mode == 'train':
                            _, l, pre, y, er = sess.run([model.train_op, model.loss,
                                model.predictions, model.y, model.error_rate],
                                feed_dict=feedDict)

                            batch_errors[batch] = er
                            print('\n{} mode, total:{},subdir:{}/{},batch:{}/{},epoch:{}/{},train loss={:.3f},mean train CER={:.3f}\n'.format(
                                FLAGS.level, total_n, id_dir+1, len(feature_dirs), batch+1, len(batched_random_idx), epoch+1, FLAGS.num_epochs, l, er/FLAGS.batch_size))

                        elif FLAGS.mode == 'dev':
                            l, pre, y, er = sess.run([model.loss, model.predictions, 
                                model.y, model.error_rate], feed_dict=feedDict)
                            batch_errors[batch] = er
                            print('\n{} mode, total:{},subdir:{}/{},batch:{}/{},dev loss={:.3f},mean dev CER={:.3f}\n'.format(
                                FLAGS.level, total_n, id_dir+1, len(feature_dirs), batch+1, len(batched_random_idx), l, er/FLAGS.batch_size))

                        elif FLAGS.mode == 'test':
                            l, pre, y, er = sess.run([model.loss, model.predictions, 
                                model.y, model.error_rate], feed_dict=feedDict)
                            batch_errors[batch] = er
                            print('\n{} mode, total:{},subdir:{}/{},batch:{}/{},test loss={:.3f},mean test CER={:.3f}\n'.format(
                                FLAGS.level, total_n, id_dir+1, len(feature_dirs), batch+1, len(batched_random_idx), l, er/FLAGS.batch_size))
                    elif FLAGS.level=='seq2seq':
                        raise ValueError('level %s is not supported now'%str(FLAGS.level))


                    # NOTE:
                    if er / FLAGS.batch_size == 1.0:
                        break

                    if batch % 20 == 0:
                        print('Truth:\n' + output_to_sequence(y, level=FLAGS.level))
                        print('Output:\n' + output_to_sequence(pre, level=FLAGS.level))

                
                    if FLAGS.mode=='train' and ((epoch * len(batched_random_idx) + batch + 1) % 20 == 0 or (
                        epoch == FLAGS.num_epochs - 1 and batch == len(batched_random_idx) - 1)):
                        checkpoint_path = os.path.join(savedir, 'model.ckpt')
                        model.saver.save(sess, checkpoint_path, global_step=epoch)
                        print('Model has been saved in {}'.format(savedir))

                end = time.time()
                delta_time = end - start
                print('Epoch ' + str(epoch + 1) + ' needs time:' + str(delta_time) + ' s')

                if FLAGS.mode=='train':
                    if (epoch + 1) % 1 == 0:
                        checkpoint_path = os.path.join(savedir, 'model.ckpt')
                        model.saver.save(sess, checkpoint_path, global_step=epoch)
                        print('Model has been saved in {}'.format(savedir))
                    epoch_er = batch_errors.sum() / total_n
                    print('Epoch', epoch + 1, 'mean train error rate:', epoch_er)

                if FLAGS.mode=='test' or FLAGS.mode=='dev':
                    with open(os.path.join(resultdir, FLAGS.level + '_result.txt'), 'a') as result:
                        result.write(output_to_sequence(y, level=FLAGS.level) + '\n')
                        result.write(output_to_sequence(pre, level=FLAGS.level) + '\n')
                        result.write('\n')
                    epoch_er = batch_errors.sum() / total_n
                    print(' test error rate:', epoch_er)



if __name__ == '__main__':
  app.run()
