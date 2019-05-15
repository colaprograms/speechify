from data import Dataset

d = Dataset()
def index(path):
    print("Indexing", path)
    d.index(path)
index(r"c:\users\meta\documents\speechify\LibriSpeech\train-clean-100")
index(r"c:\users\meta\documents\speechify\LibriSpeech\train-clean-360")
d.time()
d.dump()