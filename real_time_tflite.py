"""
#!/usr/bin/python3
#-*- coding: utf-8 -*-
@FileName: real_time_tflite.py
@Time: 2022/10/19 13:44        
@Author: zql
"""
import copy
import soundfile as sf
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import librosa


def mk_mask_mag(x):
    '''
    magnitude mask
    '''
    [noisy_real, noisy_imag, mag_mask] = x

    enh_mag_real = noisy_real * mag_mask
    enh_mag_imag = noisy_imag * mag_mask
    return enh_mag_real, enh_mag_imag


def mk_mask_pha(x):
    '''
    phase mask
    '''
    [enh_mag_real, enh_mag_imag, pha_cos, pha_sin] = x

    enh_real = enh_mag_real * pha_cos - enh_mag_imag * pha_sin
    enh_imag = enh_mag_real * pha_sin + enh_mag_imag * pha_cos

    return enh_real, enh_imag


start = time.time()

block_len = 512
block_shift = 256

# load models
interpreter = tflite.Interpreter(model_path='pretrained_weights/model_tflite/dpcrn_quant_20epochs.tflite')
interpreter.allocate_tensors()
#
# # Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# create states for the lstms
inp = np.zeros([1, 1, 257, 3], dtype=np.float32)
states = np.zeros(input_details[1]['shape'], dtype=np.float32)
# load audio file at 16k fs (please change)
win = np.sin(np.arange(.5, block_len - .5 + 1) / block_len * np.pi)
audio, fs = librosa.load('test_audio/clnsp1_train_69005_1_snr15_tl-21_fileid_158.wav',sr=16000)
# check for sampling rate
if fs != 16000:
    raise ValueError('This model only supports 16k sampling rate.')
# preallocate output audio
out_file = np.zeros((len(audio)))
# create bufferbidirectional
in_buffer = np.zeros((block_len)).astype('float32')
out_buffer = np.zeros((block_len)).astype('float32')
# calculate number of blocks
num_blocks = (audio.shape[0] - (block_len - block_shift)) // block_shift
time_array = []
out_array = []
# iterate over the number of blcoks
for idx in range(num_blocks):
    start_time = time.time()
    # shift values and write to buffer
    in_buffer = audio[idx * block_shift:(idx * block_shift) + block_len]
    # calculate fft of input block
    audio_buffer = in_buffer * win
    spec = np.fft.rfft(audio_buffer).astype('complex64')
    spec1 = copy.copy(spec)
    inp[0, 0, :, 0] = spec1.real
    inp[0, 0, :, 1] = spec1.imag
    inp[0, 0, :, 2] = 2 * np.log(abs(spec))

    # set tensors to the model
    interpreter.set_tensor(input_details[1]['index'], states)
    interpreter.set_tensor(input_details[0]['index'], inp)
    # run calculation
    interpreter.invoke()
    # get the output of the model
    output_mask = interpreter.get_tensor(output_details[0]['index'])
    output_cos = interpreter.get_tensor(output_details[1]['index'])
    output_sin = interpreter.get_tensor(output_details[2]['index'])
    states = interpreter.get_tensor(output_details[3]['index'])

    # calculate the ifft
    estimated_real, estimated_imag = mk_mask_mag([spec.real, spec.imag, output_mask])
    enh_real, enh_imag = mk_mask_pha([estimated_real, estimated_imag, output_cos, output_sin])
    estimated_complex = enh_real + 1j * enh_imag
    estimated_block = np.fft.irfft(estimated_complex)
    estimated_block = estimated_block * win
    # write block to output file
    out_file[block_shift * idx: block_shift * idx + block_len] += np.squeeze(estimated_block)
    time_array.append(time.time() - start_time)

# write to .wav file
sf.write('reco.wav', out_file, fs)
print('Processing Time [ms]:')
print(np.mean(np.stack(time_array)) * 1000)
print(time.time() - start)
print('Processing finished.')