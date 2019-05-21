from util.spectrogram_generator import Params, generator
from util.mic_display import MicrophoneDisplayer
m = MicrophoneDisplayer(44100, 128, add_deltafeatures = True)
m.start()
