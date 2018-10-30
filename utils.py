import os
import numpy as np
import tensorflow as tf


def list_to_sparse_tensor(targetList, level):
  """ turn 2-D List to SparseTensor
    """
  indices = []  #index
  vals = []  #value
  assert level == 'phn' or level == 'cha', ('type must be phoneme or character,'
                                            ' seq2seq will be supported in '
                                            'future')
  phn = [
      'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch',
      'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey',
      'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl',
      'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's',
      'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh'
  ]

  mapping = {
      'ah': 'ax',
      'ax-h': 'ax',
      'ux': 'uw',
      'aa': 'ao',
      'ih': 'ix',
      'axr': 'er',
      'el': 'l',
      'em': 'm',
      'en': 'n',
      'nx': 'n',
      'eng': 'ng',
      'sh': 'zh',
      'hv': 'hh',
      'bcl': 'h#',
      'pcl': 'h#',
      'dcl': 'h#',
      'tcl': 'h#',
      'gcl': 'h#',
      'kcl': 'h#',
      'q': 'h#',
      'epi': 'h#',
      'pau': 'h#'
  }

  group_phn = [
      'ae', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'er',
      'ey', 'f', 'g', 'h#', 'hh', 'ix', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng',
      'ow', 'oy', 'p', 'r', 's', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh'
  ]

  mapping = {
      'ah': 'ax',
      'ax-h': 'ax',
      'ux': 'uw',
      'aa': 'ao',
      'ih': 'ix',
      'axr': 'er',
      'el': 'l',
      'em': 'm',
      'en': 'n',
      'nx': 'n',
      'eng': 'ng',
      'sh': 'zh',
      'hv': 'hh',
      'bcl': 'h#',
      'pcl': 'h#',
      'dcl': 'h#',
      'tcl': 'h#',
      'gcl': 'h#',
      'kcl': 'h#',
      'q': 'h#',
      'epi': 'h#',
      'pau': 'h#'
  }

  group_phn = [
      'ae', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'er',
      'ey', 'f', 'g', 'h#', 'hh', 'ix', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng',
      'ow', 'oy', 'p', 'r', 's', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh'
  ]

  if level == 'cha':
    for tI, target in enumerate(targetList):
      for seqI, val in enumerate(target):
        indices.append([tI, seqI])
        vals.append(val)
    shape = [len(targetList), np.asarray(indices).max(axis=0)[1] + 1]  #shape
    return np.array(indices), np.array(vals), np.array(shape)

  elif level == 'phn':
    """
        for phn level, we should collapse 61 labels into 39 labels before
        scoring

        Reference:
          Heterogeneous Acoustic Measurements and Multiple Classifiers for
          Speech Recognition(1986),
            Andrew K. Halberstadt,
            https://groups.csail.mit.edu/sls/publications/1998/phdthesis-drew.pdf
        """
    for tI, target in enumerate(targetList):
      for seqI, val in enumerate(target):
        if val < len(phn) and (phn[val] in mapping.keys()):
          val = group_phn.index(mapping[phn[val]])
        indices.append([tI, seqI])
        vals.append(val)
    shape = [len(targetList), np.asarray(indices).max(0)[1] + 1]  #shape
    return np.array(indices), np.array(vals), np.array(shape)

  else:
    ##support seq2seq in future here
    raise ValueError('Invalid level: %s' % str(level))


def output_to_sequence(lmt, level='phn'):
  """ convert the output into sequences of characters or phonemes
    """
  phn = [
      'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch',
      'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey',
      'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl',
      'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's',
      'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh'
  ]
  sequences = []
  start = 0
  sequences.append([])
  for i in range(len(lmt[0])):
    if lmt[0][i][0] == start:
      sequences[start].append(lmt[1][i])
    else:
      start = start + 1
      sequences.append([])

  #here, we only print the first sequence of batch
  indexes = sequences[0]  #here, we only print the first sequence of batch
  if level == 'phn':
    seq = []
    for ind in indexes:
      if ind == len(phn):
        pass
      else:
        seq.append(phn[ind])
    seq = ' '.join(seq)
    return seq

  elif level == 'cha':
    seq = []
    for ind in indexes:
      if ind == 0:
        seq.append(' ')
      elif ind == 27:
        seq.append("'")
      elif ind == 28:
        pass
      else:
        seq.append(chr(ind + 96))
    seq = ''.join(seq)
    return seq
  else:
    raise TypeError('mode should be phoneme or character')


def get_edit_distance(hyp_arr, truth_arr, normalize, level):
  """ calculate edit distance
    This is very universal, both for cha-level and phn-level
    """

  graph = tf.Graph()
  with graph.as_default():
    truth = tf.sparse_placeholder(tf.int32)
    hyp = tf.sparse_placeholder(tf.int32)
    editDist = tf.reduce_sum(tf.edit_distance(hyp, truth, normalize=normalize))

  with tf.Session(graph=graph) as session:
    truthTest = list_to_sparse_tensor(truth_arr, level)
    hypTest = list_to_sparse_tensor(hyp_arr, level)
    feedDict = {truth: truthTest, hyp: hypTest}
    dist = session.run(editDist, feed_dict=feedDict)
  return dist


def data_lists_to_batches(inputList, targetList, batchSize, level):
  """ padding the input list to a same dimension, integrate all data into batchInputs

    """
  assert len(inputList) == len(targetList)
  # dimensions of inputList:batch*39*time_length

  nFeatures = inputList[0].shape[0]
  maxLength = 0
  for inp in inputList:
    # find the max time_length
    maxLength = max(maxLength, inp.shape[1])

  # randIxs is the shuffled index from range(0,len(inputList))
  randIxs = np.random.permutation(len(inputList))
  start, end = (0, batchSize)
  dataBatches = []

  while end <= len(inputList):
    # batchSeqLengths store the time-length of each sample in a mini-batch
    batchSeqLengths = np.zeros(batchSize)

    # randIxs is the shuffled index of input list
    for batchI, origI in enumerate(randIxs[start:end]):
      batchSeqLengths[batchI] = inputList[origI].shape[-1]

    batchInputs = np.zeros((maxLength, batchSize, nFeatures))
    batchTargetList = []
    for batchI, origI in enumerate(randIxs[start:end]):
      # padSecs is the length of padding
      padSecs = maxLength - inputList[origI].shape[1]
      # numpy.pad pad the inputList[origI] with zeos at the tail
      batchInputs[:, batchI, :] = np.pad(
          inputList[origI].T, ((0, padSecs), (0, 0)),
          'constant',
          constant_values=0)
      # target label
      batchTargetList.append(targetList[origI])
    dataBatches.append((batchInputs,
                        list_to_sparse_tensor(batchTargetList, level),
                        batchSeqLengths))
    start += batchSize
    end += batchSize
  return (dataBatches, maxLength)


def load_batched_data(mfccPath, labelPath, batchSize, level):
  """returns 3-element tuple: batched data (list), maxTimeLength (int), and
       total number of samples (int)"""

  def _read_data(path):
    with tf.gfile.GFile(path, 'r') as f:
      print('loading {}'.format(path))
      return np.load(f)

  return data_lists_to_batches([
      _read_data(os.path.join(mfccPath, fn))
      for fn in tf.gfile.ListDirectory(mfccPath)
  ], [
      _read_data(os.path.join(labelPath, fn))
      for fn in tf.gfile.ListDirectory(labelPath)
  ], batchSize, level) + (len(tf.gfile.ListDirectory(mfccPath)),)
def count_params(model, mode='trainable'):
    ''' count all parameters of a tensorflow graph
    '''
    if mode == 'all':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_op])
    elif mode == 'trainable':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_trainable_op])
    else:
        raise TypeError('mode should be all or trainable.')
    print('number of '+mode+' parameters: '+str(num))
    return num
def check_path_exists(path):
    """ check a path exists or not
    """
    if isinstance(path, list):
        for p in path:
            if not os.path.exists(p):
                os.makedirs(p)
    else:
        if not os.path.exists(path):
            os.makedirs(path)