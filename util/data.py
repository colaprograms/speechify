import sys
import os
from os.path import join
import pickle
import random
import soundfile as sf

PATH = "index"

def listdir(a, *p):
    return os.listdir(join(a, *p))

class LibriSpeech:
    """Typical use:

    ls = LibriSpeech()
    ls.load() # load from index
    batch = [ls.uniform_random() for i in range(256)]
    for file in batch:
        print(file['path'], "Duration:", file['time'], "sec")
        print("Transcript:")
        print(file['trans'])

    To make an index, see createindex.py.
    """
    def __init__(self):
        self.data = {}
        self.info = {}

    def load(self, path=PATH):
        f = open(path, "rb")
        self.data, self.info = pickle.load(f)

    def dump(self, path=PATH):
        f = open(path, "wb")
        pickle.dump([self.data, self.info], f)

    def make(self, path):
        totaltime = 0
        for reader in listdir(path):
            readerint = int(reader)
            self.data[reader] = {}
            for book in listdir(path, reader):
                bookint = int(book)
                self.data[reader][book] = []
                files = sorted(listdir(path, reader, book))
                # check for the existence of transcript
                prefix = "%d-%d" % (readerint, bookint)
                assert files[-1] == prefix + ".trans.txt"
                trans = open(join(path, reader, book, files[-1]), "r")
                # check that the rest of the
                # files are numbered and correspond to
                # the lines in the transcript file
                for i in range(0, len(files) - 1):
                    assert files[i] == prefix + "-%04d.flac" % i
                    line = trans.readline().strip().split(" ", 1)
                    assert line[0] == prefix + "-%04d" % i
                    info = sf.info(join(path, reader, book, files[i]))
                    self.data[reader][book].append({
                        'path': join(reader, book, files[i]),
                        'trans': line[1],
                        'time': info.duration,
                        'rate': info.samplerate,
                        'type': info.subtype,
                        'format': info.format
                    })
                    totaltime += info.duration
                # the transcript file should be over now
                assert trans.readline().strip() == ""
            print("Time: %f sec" % totaltime)
        cumulativetime = 0
        for r in self.data.values():
            for b in r.values():
                for f in b:
                    cumulativetime += f['time']
        self.info['totaltime'] = cumulativetime

    def train_test_split(self):
        # split readers worth about 10% of the time into a set
        # LibriSpeech has a better testing set which is more careful
        # not to overlap with the training set, but we want to
        # reserve it for special occasions, so this is our validation set
        limit = self.info['totaltime'] * 0.1
        curtime = 0
        candidates = list(self.data.keys())
        random.shuffle(candidates)
        for i in range(len(candidates)):
            reader = candidates[i]
            if curtime > limit:
                break
            for book in self.data[reader]:
                for file in self.data[reader][book]:
                    curtime += file['time']
        print("%d testing candidates, total time %f sec" % (i, curtime))
        def to_files(readerlist):
            list = []
            for reader in readerlist:
                for book in self.data[reader]:
                    for i in range(len(self.data[reader][book])):
                        list.append([reader, book, i])
            return list
        self.info['test'] = to_files(candidates[:i])
        self.info['train'] = to_files(candidates[i:])

    def uniform_test(self):
        reader, book, i = random.choice(self.info['test'])
        return self.data[reader][book][i]

    def uniform_train(self):
        reader, book, i = random.choice(self.info['train'])
        return self.data[reader][book][i]
