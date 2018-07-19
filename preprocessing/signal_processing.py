import numpy
import math


def audio2frame(signal,
                frame_length,
                frame_step,
                winfunc=lambda x: numpy.ones((x,))):
  """ Framing audio signal.

  Uses numbers of samples as unit.
    Args:
    signal: 1-D numpy array.
        frame_length: In this situation, frame_length=samplerate*win_length,
          since we use numbers of samples as unit.
    frame_step:In this situation, frame_step=samplerate*win_step, representing
      the number of samples between the start point of adjacent frames.
        winfunc:lambda function, to generate a vector with shape (x,) filled
          with ones.

    Returns:
        frames*win: 2-D numpy array with shape (frames_num, frame_length).
    """
  signal_length = len(signal)
  # Use round() to ensure length and step are integer, considering that we use numbers
  # of samples as unit.
  frame_length = int(round(frame_length))
  frame_step = int(round(frame_step))
  if signal_length <= frame_length:
    frames_num = 1
  else:
    frames_num = 1 + int(
        math.ceil((1.0 * signal_length - frame_length) / frame_step))
  pad_length = int((frames_num - 1) * frame_step + frame_length)
  # Padding zeros at the end of signal if pad_length > signal_length.
  zeros = numpy.zeros((pad_length - signal_length,))
  pad_signal = numpy.concatenate((signal, zeros))
  # Calculate the indice of signal for every sample in frames, shape (frams_nums, frams_length)
  indices = numpy.tile(numpy.arange(0, frame_length),
                       (frames_num, 1)) + numpy.tile(
                           numpy.arange(0, frames_num * frame_step, frame_step),
                           (frame_length, 1)).T
  indices = numpy.array(indices, dtype=numpy.int32)
  # Get signal data according to indices.
  frames = pad_signal[indices]
  win = numpy.tile(winfunc(frame_length), (frames_num, 1))
  return frames * win


def deframesignal(frames,
                  signal_length,
                  frame_length,
                  frame_step,
                  winfunc=lambda x: numpy.ones((x,))):
  signal_length = round(signal_length)
  frame_length = round(frame_length)
  frames_num = numpy.shape(frames)[0]
  assert numpy.shape(
      frames
  )[1] == frame_length, 'col of frames matrix should be equal to a frame length'
  indices = numpy.tile(numpy.arange(0, frame_length),
                       (frames_num, 1)) + numpy.tile(
                           numpy.arange(0, frames_num * frame_step, frame_step),
                           (frame_length, 1)).T
  indices = numpy.array(indices, dtype=numpy.int32)
  pad_length = (frames_num - 1) * frame_step + frame_length
  if signal_length <= 0:
    signal_length = pad_length
  recalc_signal = numpy.zeros((pad_length,))
  window_correction = numpy.zeros((pad_length, 1))
  win = winfunc(frame_length)
  for i in range(0, frames_num):
    window_correction[indices[
        i, :]] = window_correction[indices[i, :]] + win + 1e-15
    recalc_signal[indices[i, :]] = recalc_signal[indices[i, :]] + frames[i, :]
  recalc_signal = recalc_signal / window_correction
  return recalc_signal[0:signal_length]


def spectrum_magnitude(frames, NFFT):
  """Apply FFT and Calculate magnitude of the spectrum.

    Args:
        frames: 2-D frames array calculated by audio2frame(...).
        NFFT:FFT size.

    Returns:
        Return magnitude of the spectrum after FFT, with shape (frames_num,
        NFFT).
    """
  complex_spectrum = numpy.fft.rfft(frames, NFFT)
  return numpy.absolute(complex_spectrum)


def spectrum_power(frames, NFFT):
  """Calculate power spectrum for every frame after FFT.
    Args:
        frames: 2-D frames array calculated by audio2frame(...).
        NFFT:FFT size
    Returns:
        Power spectrum: PS = magnitude^2/NFFT
    """
  return 1.0 / NFFT * numpy.square(spectrum_magnitude(frames, NFFT))


def log_spectrum_power(frames, NFFT, norm=1):
  """Calculate log power spectrum.
    Args:
        frames:2-D frames array calculated by audio2frame(...)
        NFFT:FFT size
        norm: Norm.
    """
  spec_power = spectrum_power(frames, NFFT)
  # In case of calculating log0, we set 0 in spec_power to 0.
  spec_power[spec_power < 1e-30] = 1e-30
  log_spec_power = 10 * numpy.log10(spec_power)
  if norm:
    return log_spec_power - numpy.max(log_spec_power)
  else:
    return log_spec_power


def pre_emphasis(signal, coefficient=0.95):
  """Pre-emphasis.
    Args:
        signal: 1-D numpy array.
        coefficient:Coefficient for pre-emphasis. Defauted to 0.95.
    Returns:
        pre-emphasis signal.
    """
  return numpy.append(signal[0], signal[1:] - coefficient * signal[:-1])
