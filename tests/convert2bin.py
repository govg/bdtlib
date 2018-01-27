import os
import sys

filename_read = '../data/ml-1m/ratings_time.dat'
filename_write = '../data/ml-1m/ratings_time_binary.dat'
fp = open(filename_read, 'r')
fp2 = open(filename_write, 'w')
line = fp.readline()
while line:
	temp = line.strip('\n').split('::')	
	user_id = temp[0]
	movie_id = temp[1]
	true_rating = int(temp[2])
	timestamp = temp[3]
	if true_rating >= 3.5:
		true_rating = str(1)
	else:
		true_rating = str(0)

	fp2.write(user_id + '::' + movie_id + '::' + true_rating + '::' + timestamp + '\n')
	line = fp.readline()

fp.close()
fp2.close()
