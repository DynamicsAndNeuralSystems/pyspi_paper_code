import os
import _pickle as cPickle

try:
    from pyspi.calculator import CalculatorFrame, CorrelationFrame
except ModuleNotFoundError:
    print("Note: Using legacy code.")
    from pynats.calculator import CalculatorFrame, CorrelationFrame

dbfile = "data/database.pkl"
if not os.path.exists(dbfile):
    import urllib.request

    question = f"File \"{dbfile}\" does not exist. Shall I download it now? [Y/n] "
    reply = str(input(question)).lower().strip()

    if reply == 'y':
        print("Downloading now. This may take some time...")
        URL = "https://zenodo.org/record/7113213/files/database.pkl?download=1"
        urllib.request.urlretrieve(URL, dbfile)
        print("Done.")
    else:
        print("OK, Exiting.")
        exit(0)

with open(dbfile, "rb") as f:
    database = cPickle.load(f)

names = list(database.keys())

# To test
names = names[:3]

datasets = [database[n]['data'].T for n in names]
labels = [database[n]['labels'] for n in names]

# Set fast=True to speed things up
try:
    calcf = CalculatorFrame(datasets=datasets, labels=labels, names=names, fast=True)
except TypeError:
    calcf = CalculatorFrame(datasets=datasets, labels=labels, names=names)
calcf.compute()

corrf = CorrelationFrame(calcf)
mm_adj = corrf.get_average_correlation()
print(mm_adj)