import os
import glob
import scipy.signal
import scipy.fftpack
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import random
import math


def readFile(filepath):
    rat, wav = scipy.io.wavfile.read(filepath)
    duratio = len(wav) / rat
    tim = np.arange(0, duratio, 1 / rat)
    return rat, wav, duratio, tim


def openFile(path):
    data = []
    folder_data = glob.glob(path + "\\*.wav")
    for count, wavFile in enumerate(folder_data):
        rate, wave, duration, timeinsec = readFile(wavFile)
        data.append(wave)
    annotation = []
    folder_annotation = glob.glob(path + "\\*.txt")
    for count, txtFile in enumerate(folder_annotation):
        array = np.loadtxt(txtFile)
        annotation.append(array)
    return data, annotation


# Assignment 1 acf
def block_audio(x, blockSize, hopSize, fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)), axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return xb, t


def comp_acf(inputVector, bIsNormalized=True):
    if bIsNormalized:
        norm = np.dot(inputVector, inputVector)
    else:
        norm = 1
    afCorr = np.correlate(inputVector, inputVector, "full") / norm
    afCorr = afCorr[np.arange(inputVector.size - 1, afCorr.size)]
    return afCorr


def get_f0_from_acf(r, fs):
    eta_min = 1
    afDeltaCorr = np.diff(r)
    eta_tmp = np.argmax(afDeltaCorr > 0)
    eta_min = np.max([eta_min, eta_tmp])
    f = np.argmax(r[np.arange(eta_min + 1, r.size)])
    f = fs / (f + eta_min + 1)
    return f


def track_pitch_acf(x, blockSize, hopSize, fs):
    # get blocks
    [xb, t] = block_audio(x, blockSize, hopSize, fs)
    # init result
    f0 = np.zeros(xb.shape[0])
    # compute acf
    for n in range(0, xb.shape[0]):
        r = comp_acf(xb[n, :])
        f0[n] = get_f0_from_acf(r, fs)
    return f0, t


# Maximum spectral peak based pitch tracker
def compute_spectrogram(xb, fs):
    fInHz = []
    for i, value in enumerate(xb):
        xb[i] = value * scipy.signal.hann(np.shape(value)[0])
    xb = scipy.fftpack.fft(xb)
    spec = np.absolute(np.vstack(xb))
    for frame in spec:
        fInHz.append(frame.tolist().index(max(frame.tolist())) * fs / np.shape(xb)[1])
    return spec, fInHz


#def compute_spectrogram(xb, fs):
#    """Calculate the short-time Fourier transform magnitude.
#
#    Args:
#    xb: blocked audio obtained from block_audio.
#    fs: sampling frequency.
#
#    Returns:
#    a matrix where each row contains the magnitudes of the fft_length/2+1
#    unique values of the FFT for the corresponding frame of input samples and fInHertz.
#    """
#
 #   def periodic_hann(window_length):
 #       """Calculate a "periodic" Hann window.
 #       window_length: The number of points in the returned window.
#
#        Returns:
#        A 1D np.array containing the periodic hann window.
#        """
#        return 0.5 - (0.5 * np.cos(2 * np.pi / window_length *
#                                   np.arange(window_length)))

#    s = fs
#    freq_max = 2000
#    r_min = int(round(fs / freq_max)) - 1
#    window = periodic_hann(np.shape(xb)[1])
#    print(window.shape)
#    print(xb[0].shape)
#    X = np.abs(np.fft.rfft(xb[0] * window))
#    for block in xb[1:]:
#        windowed_frames = block * window
#        X = np.vstack([X, np.abs(np.fft.rfft(windowed_frames))])
#
 #   #     if (len(X[:, scipy.signal.find_peaks(X[:, 0])[0]][0])):
#    f = np.amax(X[:, scipy.signal.find_peaks(X[:, 0])[0] % X.shape[1]])
#    #     else:
#    #     f = np.amax(X[:, 0])
#
#    for i in range(1, len(X[0])):
#        #         if (len(X[:, scipy.signal.find_peaks(X[:, i])[0]][0])):
 #       f = np.vstack([f, np.amax(X[:, scipy.signal.find_peaks(X[:, i])[0] % X.shape[1]])])
 #   #         else:
 #   #         f = np.vstack([f, np.amax(X[:, i])])
#
#    fInHz = s * fs / (f + r_min + 1)
#    return X, fInHz.reshape(int(xb.shape[1] / 2 + 1), )


def track_pitch_fftmax(x, blockSize, hopSize, fs):
    # estimates the fundamental frequency f0 based on block-wise max. spectral peak finding approach
    xb, timeInSec = block_audio(x, blockSize, hopSize, fs)
    X, fInHz = compute_spectrogram(xb, fs)

    #     f0 = max(X[:,0][scipy.signal.find_peaks(X[:,0])[0]])
    #     for row in X[1:]:
    #         f0 = np.vstack([f0, max(row[scipy.signal.find_peaks(row)[0]])])
    f0 = fInHz
    return f0, timeInSec


# HPS (Harmonic Product Spectrum) based pitch tracker
def get_f0_from_Hps(X, fs, order):
    # initialize
    X = X.T
    f_min = 300
    f = np.zeros(X.shape[1])

    iLen = int((X.shape[0] - 1) / order)
    afHps = X[np.arange(0, iLen), :]
    k_min = int(round(f_min / fs * 2 * (X.shape[0] - 1)))

    # compute the HPS
    for j in range(1, order):
        X_d = X[::(j + 1), :]
        afHps *= X_d[np.arange(0, iLen), :]

    f = np.argmax(afHps[np.arange(k_min, afHps.shape[0])], axis=0)

    # find max index and convert to Hz
    f = (f + k_min) / (X.shape[0] - 1) * fs / 2

    return f
    #"""Estimate frequency using harmonic product spectrum
    #"""
    #f_min = 500
    #f = np.zeros(X.shape[1])
    #r_min = int(round(f_min / fs * 2 * (X.shape[0] - 1)))
#
    ## decimate the len based on the order
    #compressed_len = int((X.shape[0] - 1) / order)

    #hps = X[np.arange(0, compressed_len), :]
    #for i in range(1, order):
    #    # step by i+1
    #    X_decimated = X[::(i + 1), :]
    #    hps *= X_decimated[np.arange(0, compressed_len), :]

    #f = np.argmax(hps[np.arange(r_min, hps.shape[0])], axis=0)

    ## find max index and convert to Hz
    #f = 0.5 * fs * (f + r_min) / (X.shape[0] - 1)

    #return f


def track_pitch_hps(x, blockSize, hopSize, fs):
    xb, timeInSec = block_audio(x, blockSize, hopSize, fs)
    X, fInHz = compute_spectrogram(xb, fs)
    f0 = get_f0_from_Hps(X, fs=44100, order=4)
    return f0, timeInSec


# voice detection
def extract_rms(xb):
    # Returns an array (NumOfBlocks X k) of spectral flux for all the audio blocks: k = frequency bins
    # xb is a matrix of blocked audio data (dimension NumOfBlocks X blockSize)

    K = len(xb)
    rms_vec = np.zeros(K)
    for i, frame in enumerate(xb):
        ms = np.dot(frame, frame) / K
        rms_vec[i] = np.sqrt(ms)

    # convert to db
    rms_Db = 20 * np.log10(rms_vec)

    # truncate to 100 dB
    rms_Db[rms_Db > 100] = 100

    return rms_Db


def create_voicing_mask(rmsDb, thresholdDb):
    for rms_dB in rmsDb:
        if rms_dB >= thresholdDb:
            rms_dB == 1
        if rms_dB < thresholdDb:
            rms_dB == 0
    return rmsDb


def apply_voicing_mask(f0, mask):
    return f0 * mask


# Different evaluation metrics
def eval_voiced_fp(estimation, annotation):
    # dataset = []
    # folder = glob.glob(path + "\\*.txt")
    # for count, txtFile in enumerate(folder):
    #    array = np.loadtxt(txtFile, dtype=str)
    #    dataset.append(array)
    denominator = 0
    numerator = 0
    for i, value in enumerate(estimation):
        if float(annotation[i][2]) == 0:
            denominator = denominator + 1
            if float(value) != 0:
                numerator = numerator + 1
    return numerator / denominator


def eval_voiced_fn(estimation, annotation):
    denominator = 0
    numerator = 0
    for i, value in enumerate(estimation):
        if float(annotation[i][2]) != 0:
            denominator = denominator + 1
            if float(value) == 0:
                numerator = numerator + 1
    return numerator / denominator


def eval_pitchtrack_v2(estimation, annotation):
    errCentRms = []
    for i, value in enumerate(estimation):
        estimateInHz = value
        if annotation[i][2] == 0:
            errCentRms.append(0)
        else:
            groundtruthInHz = annotation[i][2]
            if estimateInHz == 0:
                errCentRms.append(0)
            else:
                errCentRms.append(1200 * math.log(estimateInHz / groundtruthInHz, 2))
    errCentRms = np.array(errCentRms)
    pfp = eval_voiced_fp(estimation, annotation)
    pfn = eval_voiced_fn(estimation, annotation)
    return errCentRms, pfp, pfn


#Evaluation
def sin_wave(A, f, fs, phi, t):
    Ts = 1/fs
    n = t / Ts
    n = np.arange(n)
    y = A*np.sin(2*np.pi*f*n*Ts + phi*(np.pi/180))
    return y


def executeassign3():
    sinwave = sin_wave(A=1, f=441, fs=44100, phi=0, t=1)
    sinwave = np.hstack((sinwave, sin_wave(A=1, f=882, fs=44100, phi=0, t=1)))

    #E.1
    f0_fft, timeInSec_fft = track_pitch_fftmax(sinwave, 1024, 512, 44100)
    f0_hps, timeInSec_hps = track_pitch_hps(sinwave, 1024, 512, 44100)
    y_fft = f0_fft
    x_fft = timeInSec_fft
    y_hps = f0_hps
    x_hps = timeInSec_hps
    plt.subplot(2, 2, 1)
    plt.title("f0 of fftmax")
    #plt.plot(x_fft, y_fft, label=plotName)
    plt.plot(y_fft)
    plt.subplot(2, 2, 2)
    plt.title("absolute error of fftmax")
    abserror_fft = []
    for i, value in enumerate(f0_fft):
        if i<=np.shape(f0_fft)[0]/2:
            abserror_fft.append(abs(value-441))
        else:
            abserror_fft.append(abs(value - 882))
    abserror_fft = np.array(abserror_fft)
    plt.plot(abserror_fft)
    plt.subplot(2, 2, 3)
    plt.title("f0 of hps")
    #plt.plot(x_hps, y_hps, label=plotName)
    plt.plot(y_hps)
    plt.subplot(2, 2, 4)
    plt.title("absolute error of hps")
    abserror_hps = []
    for i, value in enumerate(f0_hps):
        if i <= np.shape(f0_hps)[0] / 2:
            abserror_hps.append(abs(value - 441))
        else:
            abserror_hps.append(abs(value - 882))
    abserror_hps = np.array(abserror_hps)
    plt.plot(abserror_hps)
    plt.legend(loc='upper right', fontsize=10)
    plt.show()

    #E.2
    f0_fft_2, timeInSec_fft_2 = track_pitch_fftmax(sinwave, 2048, 512, 44100)
    y_fft_2 = f0_fft_2
    x_fft_2 = timeInSec_fft_2
    plt.subplot(1, 2, 1)
    plt.title("f0 of fftmax")
    plt.plot(y_fft_2)
    plt.subplot(1, 2, 2)
    plt.title("absolute error of fftmax")
    abserror_fft_2 = []
    for i, value in enumerate(f0_fft_2):
        if i <= np.shape(f0_fft_2)[0] / 2:
            abserror_fft_2.append(abs(value - 441))
        else:
            abserror_fft_2.append(abs(value - 882))
    abserror_fft_2 = np.array(abserror_fft_2)
    plt.plot(abserror_fft_2)
    plt.legend(loc='upper right', fontsize=10)
    plt.show()

    #E.3
    errCentRms = []
    pfp = []
    pfn = []
    data, annotation = openFile("D:\SchoolWork\ACA\ACAassign3/trainData")
    for i, wav in enumerate(data):
        f0, timeInSec = track_pitch_fftmax(wav, 1024, 512, 44100)
        errCentRms_perFile, pfp_perFile, pfn_perFile = eval_pitchtrack_v2(f0, annotation[i])
        errCentRms.append(errCentRms_perFile)
        pfp.append(pfp_perFile)
        pfn.append(pfn_perFile)
    average_errCentRms = []
    for ecr in errCentRms:
        average_errCentRms.append(sum(ecr)/len(ecr))
    average_pfp = sum(pfp) / 3
    average_pfn = sum(pfn) / 3
    return average_errCentRms, average_pfp, average_pfn


def track_pitch(x, blockSize, hopSize, fs, method, voicingThres):
    if method == "max":
        f0, timeInSec = track_pitch_fftmax(x, blockSize, hopSize, fs)
    elif method == "hps":
        f0, timeInSec = track_pitch_hps(x, blockSize, hopSize, fs)
    elif method == "acf":
        f0, timeInSec = track_pitch_acf(x, blockSize, hopSize, fs)
    f0 = apply_voicing_mask(f0, create_voicing_mask(extract_rms(block_audio(x, blockSize, hopSize, fs)[0]), voicingThres))  #dose this rms have the right dimension?
    f0 = np.array(f0)
    return f0, timeInSec

executeassign3()

data, annotation = openFile("D:\SchoolWork\ACA\ACAassign3/trainData")

errCentRms = []
pfp = []
pfn = []
for i, wav in enumerate(data):
    f0, timeInSec = track_pitch_fftmax(wav, 1024, 512, 44100)
    errCentRms_perFile, pfp_perFile, pfn_perFile = eval_pitchtrack_v2(track_pitch(wav, 1024, 512, 44100, "max", -40)[0], annotation[i])
    errCentRms.append(errCentRms_perFile)
    pfp.append(pfp_perFile)
    pfn.append(pfn_perFile)
result_max_40 = np.array(errCentRms)
np.vstack((result_max_40, np.array(pfp)))
np.vstack((result_max_40, np.array(pfn)))

errCentRms = []
pfp = []
pfn = []
for i, wav in enumerate(data):
    f0, timeInSec = track_pitch_fftmax(wav, 1024, 512, 44100)
    errCentRms_perFile, pfp_perFile, pfn_perFile = eval_pitchtrack_v2(track_pitch(wav, 1024, 512, 44100, "max", -20)[0], annotation[i])
    errCentRms.append(errCentRms_perFile)
    pfp.append(pfp_perFile)
    pfn.append(pfn_perFile)
result_max_20 = np.array(errCentRms)
np.vstack((result_max_20, np.array(pfp)))
np.vstack((result_max_20, np.array(pfn)))

errCentRms = []
pfp = []
pfn = []
for i, wav in enumerate(data):
    f0, timeInSec = track_pitch_fftmax(wav, 1024, 512, 44100)
    errCentRms_perFile, pfp_perFile, pfn_perFile = eval_pitchtrack_v2(track_pitch(wav, 1024, 512, 44100, "hps", -40)[0], annotation[i])
    errCentRms.append(errCentRms_perFile)
    pfp.append(pfp_perFile)
    pfn.append(pfn_perFile)
result_hps_40 = np.array(errCentRms)
np.vstack((result_hps_40, np.array(pfp)))
np.vstack((result_hps_40, np.array(pfn)))

errCentRms = []
pfp = []
pfn = []
for i, wav in enumerate(data):
    f0, timeInSec = track_pitch_fftmax(wav, 1024, 512, 44100)
    errCentRms_perFile, pfp_perFile, pfn_perFile = eval_pitchtrack_v2(track_pitch(wav, 1024, 512, 44100, "hps", -20)[0], annotation[i])
    errCentRms.append(errCentRms_perFile)
    pfp.append(pfp_perFile)
    pfn.append(pfn_perFile)
result_hps_20 = np.array(errCentRms)
np.vstack((result_hps_20, np.array(pfp)))
np.vstack((result_hps_20, np.array(pfn)))

errCentRms = []
pfp = []
pfn = []
for i, wav in enumerate(data):
    f0, timeInSec = track_pitch_fftmax(wav, 1024, 512, 44100)
    errCentRms_perFile, pfp_perFile, pfn_perFile = eval_pitchtrack_v2(track_pitch(wav, 1024, 512, 44100, "acf", -40)[0], annotation[i])
    errCentRms.append(errCentRms_perFile)
    pfp.append(pfp_perFile)
    pfn.append(pfn_perFile)
result_acf_40 = np.array(errCentRms)
np.vstack((result_acf_40, np.array(pfp)))
np.vstack((result_acf_40, np.array(pfn)))

errCentRms = []
pfp = []
pfn = []
for i, wav in enumerate(data):
    f0, timeInSec = track_pitch_fftmax(wav, 1024, 512, 44100)
    errCentRms_perFile, pfp_perFile, pfn_perFile = eval_pitchtrack_v2(track_pitch(wav, 1024, 512, 44100, "acf", -20)[0], annotation[i])
    errCentRms.append(errCentRms_perFile)
    pfp.append(pfp_perFile)
    pfn.append(pfn_perFile)
result_acf_20 = np.array(errCentRms)
np.vstack((result_acf_20, np.array(pfp)))
np.vstack((result_acf_20, np.array(pfn)))

print("end")

