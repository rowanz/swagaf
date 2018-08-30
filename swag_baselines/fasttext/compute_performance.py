import numpy as np
import argparse
import os
# neg probability, pos prob

parser = argparse.ArgumentParser(description='compute performance')
parser.add_argument('fn', metavar='fn', type=str,
                    help='filename 2 use')
fn = parser.parse_args().fn

rez = []
with open(fn, 'r') as f:
    all_lines = f.read().splitlines()

for line in all_lines:
    linesplit = line.split(' ')
    if linesplit[0] == '__label__gold':
        rez.append((float(linesplit[3]), float(linesplit[1])))
    else:
        rez.append((float(linesplit[1]), float(linesplit[3])))
rez_np = np.array(rez)[:,1].reshape((len(rez) // 4, 4))

ranks = (-rez_np).argsort(1).argsort(1)[:,0]
print("accuracy is {:.3f}".format(np.mean(ranks == 0)))

print('deleting {}'.format(fn))
os.remove(fn)