import array

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, normalize, resample, butter
from scipy.io import wavfile


def import_audio(file):
    fs, data_stereo = wavfile.read(file)

    # For stereo audio only select one channel
    data2 = data_stereo[:, 0]
    return fs, data2


def normalize_data(data):
    max_data = max(abs(data))
    normalized = data / max_data
    return normalized


def normalize_data_2(data):
    return normalize(data, 1)


def up_sample(data, factor):
    return resample(data, factor * len(data))



if __name__ == '__main__':

    # Temp vars
    factor = 10

    # AM Modulation
    A_c = 1  # Carrier amplitude
    f_c = 80e3  # Carrier frequency
    m = 0.7  # Modulation index

    feq_sam, data = import_audio('PyCharm')
    data_normalized = normalize_data(data)
    up_sampled_data = up_sample(data, factor)

    fs2 = feq_sam*factor

    t_max = len(up_sampled_data) / fs2
    t = np.linspace(0, t_max, (int)(t_max * fs2))
    carrier = np.cos(2 * np.pi * f_c * t)
    am = (A_c + A_c * m * up_sampled_data) * carrier

    # Noise
    noise_power = 0.01
    mean = 0
    std = np.sqrt(noise_power)
    noise = np.random.normal(mean, std, size=len(am))

    # Signal + noise
    am_rx = am + noise

    # Synchronous rx

    mixer_out = am_rx * carrier

    fc = fs / 2
    b, a = butter(5, fc / (fs2 * 0.5), btype='low', analog=False)
    modulator_rx_synch = filtfilt(b, a, mixer_out)
    modulator_rx_synch = modulator_rx_synch - np.mean(modulator_rx_synch)

    # Output Sound
    modulator_rx_synch = (modulator_rx_synch * max_data).astype(np.int16)
    wavfile.write("output_file.wav", fs2, modulator_rx_synch)
