import util.sample
import time
import cProfile
s = util.sample.Sampler()
file, buf = s.rand()
print(file)
util.sample.imshow(buf)

def loop(m):
    start = time.time()
    for i in range(m):
        file, buf = s.rand()
    print("%d iterations: %f" % (m, time.time() - start))

# atashi iya ne
#cProfile.run("loop(100)")
