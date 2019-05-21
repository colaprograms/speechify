# Configuration

place = "sandbox"
#place = "home"

if place == "sandbox":
    index = "index"
    path = r"c:\users\meta\documents\speechify\LibriSpeech\combined"
    microphone_volume_range = -3, 4
else:
    index = "index"
    path = "../speechify_dat/combined"
    microphone_volume_range = -4.9, 1.2

if True:
    librispeech_range = -4, 4
