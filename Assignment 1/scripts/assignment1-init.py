
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt # matplot lib is the premiere plotting lib for Python: https://matplotlib.org/
import numpy as np # numpy is the premiere signal handling library for Python: http://www.numpy.org/
import scipy as sp # for signal processing
from scipy import signal
import random
import scripts
from scripts import signal
from scripts import audio
from scripts.sample_helper import sample_frequencies
from scripts.sample_helper import sample_frequency
from scripts.sample_helper import apply_fir_filter
from scripts.sample_helper import signal_quantization


# Imports for part 2
import matplotlib.pyplot as plt
import numpy as np
from scripts.binary_generator import gen_binary_array
from math import pi
