from util.data import LibriSpeech

ls = LibriSpeech()
ls.make("../speechify_dat/combined")
ls.dump()
