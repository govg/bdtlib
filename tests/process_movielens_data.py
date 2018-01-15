import os
import sys
import numpy as np

filename_read = '../data/ml-10M100K/ratings.dat'
filename_write = '../data/ml-10M100K/ratings_time.dat'
fp = open(filename_write, 'r')

line = fp.readline()
timestamps = []
lines = []
while line:
	time_of_log = int(line.strip('\n').split('::')[3])
	timestamps.append(time_of_log)
	lines.append(line)
	line = fp.readline()

timestamps = np.array(timestamps)
lines = np.array(lines)
index = np.argsort(timestamps)
lines = lines[index]

lines = list(lines)
# fpw = open(filename_write, 'w')
# fpw.writelines(lines)

