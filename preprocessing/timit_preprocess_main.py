"""Do MFCC over all *.wav files and parse label file Use os.walk to iterate all files in a root directory original phonemes: phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh'] mapped phonemes(For more details, you can read the main page of this repo): phn = ['sil', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'el', 'en', 'epi', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'ix', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'q', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']

"""

import os
import argparse
import glob
import sys
import sklearn
import numpy as np
import scipy.io.wavfile as wav
from sklearn import preprocessing
from calc_mfcc import calcfeat_delta_delta
from spectrogram import spectrogramPower
import tensorflow as tf
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('input_path', '',
                       'Directory where Timit dataset is contained')
tf.flags.DEFINE_string('output_path', '',
                       'Directory where preprocessed arrays are to be saved')
tf.flags.DEFINE_string('split', 'train', 'Name of the dataset')
tf.flags.DEFINE_string('level', 'phn', 'character (cha) or phoneme (phn) level')
tf.flags.DEFINE_string('mode', 'mfcc', 'mfcc or fbank')
tf.flags.DEFINE_integer('featlen', 13, 'Features length')
tf.flags.DEFINE_bool('seq2seq', False, 'set this flag to use seq2seq')

tf.flags.DEFINE_float('winlen', 0.025, 'specify the window length of feature')

tf.flags.DEFINE_float('winstep', 0.01,
                      'specify the window step length of feature')

## original phonemes
phn = [
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch',
    'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey',
    'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l',
    'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh',
    't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh'
]

## cleaned phonemes
#phn = ['sil', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'el', 'en', 'epi', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'ix', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'q', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']


def wav2feature(rootdir, save_directory, mode, feature_len, level, keywords,
                win_len, win_step, seq2seq, save):
  feat_dir = os.path.join(save_directory, level, keywords, mode)
  label_dir = os.path.join(save_directory, level, keywords, 'label')
  if not tf.gfile.Exists(label_dir):
    tf.gfile.MakeDirs(label_dir)
  if not tf.gfile.Exists(feat_dir):
    tf.gfile.MakeDirs(feat_dir)
  count = 0
  for subdir, dirs, files in tf.gfile.Walk(rootdir):
    for file in files:
      fullFilename = os.path.join(subdir, file)
      filenameNoSuffix = os.path.splitext(fullFilename)[0]
      if file.endswith('.WAV'):
        rate = None
        sig = None
        try:
          with tf.gfile.GFile(fullFilename, 'r') as f:
            (rate, sig) = wav.read(f)
        except ValueError as e:
          if e.message == "File format 'NIST'... not understood.":
            print('You should use nist2wav.sh to convert NIST format files to '
                  'WAV files first.')
            return
        feat = calcfeat_delta_delta(
            sig,
            rate,
            win_length=win_len,
            win_step=win_step,
            mode=mode,
            feature_len=feature_len)
        feat = preprocessing.scale(feat)
        feat = np.transpose(feat)
        print(feat.shape)

        if level == 'phn':
          labelFilename = filenameNoSuffix + '.PHN'
          phenome = []
          with tf.gfile.GFile(labelFilename, 'r') as f:
            if seq2seq is True:
              phenome.append(len(phn))  # <start token>
            for line in f.read().splitlines():
              s = line.split(' ')[2]
              p_index = phn.index(s)
              phenome.append(p_index)
            if seq2seq is True:
              phenome.append(len(phn) + 1)  # <end token>
            print(phenome)
          phenome = np.array(phenome)

        elif level == 'cha':
          labelFilename = filenameNoSuffix + '.WRD'
          phenome = []
          sentence = ''
          with tf.gfile.GFile(labelFilename, 'r') as f:
            for line in f.read().splitlines():
              s = line.split(' ')[2]
              sentence += s + ' '
              if seq2seq is True:
                phenome.append(28)
              for c in s:
                if c == "'":
                  phenome.append(27)
                else:
                  phenome.append(ord(c) - 96)
              phenome.append(0)

            phenome = phenome[:-1]
            if seq2seq is True:
              phenome.append(29)
          print(phenome)
          print(sentence)

        count += 1
        print('file index:', count)
        if save:
          featureFilename = os.path.join(
              feat_dir,
              filenameNoSuffix.split('/')[-2] + '-' +
              filenameNoSuffix.split('/')[-1] + '.npy')
          print(featureFilename)
          with tf.gfile.GFile(featureFilename, 'w') as wf:
            np.save(wf, feat)
          labelFilename = os.path.join(
              label_dir,
              filenameNoSuffix.split('/')[-2] + '-' +
              filenameNoSuffix.split('/')[-1] + '.npy')
          print(labelFilename)
          with tf.gfile.GFile(labelFilename, 'w') as wf:
            np.save(wf, phenome)


def main(_):
  # character or phoneme
  root_directory = FLAGS.input_path
  save_directory = FLAGS.output_path
  level = FLAGS.level
  mode = FLAGS.mode
  feature_len = FLAGS.featlen
  name = FLAGS.split
  seq2seq = FLAGS.seq2seq
  win_len = FLAGS.winlen
  win_step = FLAGS.winstep

  root_directory = os.path.join(root_directory, name)
  if not tf.gfile.IsDirectory(root_directory):
    raise ValueError('Root directory {} does not exist!'.format(root_directory))
  if not tf.gfile.Exists(save_directory):
    tf.gfile.MakeDirs(save_directory)
  wav2feature(
      root_directory,
      save_directory,
      mode=mode,
      feature_len=feature_len,
      level=level,
      keywords=name,
      win_len=win_len,
      win_step=win_step,
      seq2seq=seq2seq,
      save=True)


if __name__ == '__main__':
  tf.app.run()
