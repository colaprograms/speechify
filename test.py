from data import Dataset

d = Dataset()
d.load()
for a, b in d.data.items():
    print(a)
