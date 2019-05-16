import util.sample
s = util.sample.Sampler()
file, buf = s.rand()
print("Path:", file['path'])
print("Trans:", file['trans'])
util.sample.imshow(buf)