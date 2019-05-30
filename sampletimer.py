import util.sample
import time
import cProfile

def show_one():
    file, buf = util.sample.Sampler().rand()
    print(file)
    util.sample.imshow(buf)


def loop(m):
    s = util.sample.Sampler()
    start = time.time()
    for i in range(m):
        file, buf = s.rand()
    print("%d iterations: %f" % (m, time.time() - start))

if __name__ == "__main__":
    cProfile.run("loop(100)")
