# Configuration

place = "sandbox"
#place = "home"

if place == "sandbox":
    index = "index"
    path = r"c:\users\meta\documents\speechify\LibriSpeech\combined"
    microphone_volume_range = -6, 1.2
else:
    index = "index"
    path = "../speechify_dat/combined"
    specpath = "/home/cola/src/spectrogram"#specpath = "../speechify_dat/spectrogram"
    microphone_volume_range = -6, 1.2 #-4.9, 1.2

if True:
    librispeech_range = -4, 4
