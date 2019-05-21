from util.spectrogram_generator import Params, generator
from util.mic_display import MicrophoneDisplayer
m = MicrophoneDisplayer(16000, 80, add_deltafeatures = True)
m.start()
