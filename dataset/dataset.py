import sys
import os
import os.path

class Dataset:
    def __init__(self, suffix, path = r"C:\Users\meta\Documents\speechify\LibriSpeech"):
        self.path = path
        if suffix is not None:
            self.path = os.path.join(self.path, suffix)
        self.data = {}
        
    def load(self):
        for reader in os.listdir(self.path):
            readerint = int(reader)
            _reader = {}
            self.data[reader] = _reader
            readerpath = os.join(self.path, reader)
            for book in os.listdir(readerpath):
                bookint = int(book)
                _bk = []
                reader[book] = _bk
                bookpath = os.join(readerpath, book)
                prefix = "%d-%d-" % (readerint, bookint)
                files = sorted(os.listdir(bookpath))
                assert files[0] == prefix + ".trans"
                trans = open(os.join(bookpath, files[0]), "r")
                for i in range(0, len(files) - 1):
                    assert files[i + 1] == prefix + "%04d" % index
                    line = trans.readline().strip().split(" ", 1)
                    assert line[0] == files[i+1]
                    _bk.append({
                        'path': os.join(bookpath, files[i+1]),
                        'trans': line[1]
                    })