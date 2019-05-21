from util.data import LibriSpeech

ls = LibriSpeech()
ls.make("../speechify_dat/combined")
ls.train_test_split()
ls.dump()
