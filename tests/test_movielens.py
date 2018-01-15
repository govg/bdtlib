import os
import sys
import numpy as np

def create_context_vector(user_id, user_attribute):
	vec_len = 49
	vec = np.zeros(vec_len)

	user_id_bin = bin(user_id).strip('0b')
	index = 0
	for i in range(len(user_id_bin)):
		vec[index] = int(user_id_bin[i])
		index += 1

	if gender=='M':
		vec[index] = 1
		index += 1

	age_bin = bin(age).strip('0b')
	for i in range(len(age_bin)):
		vec[index] = int(age_bin[i])
		index += 1

	occupation_bin = bin(occupation).strip('0b')
	for i in range(len(occupation_bin)):
		vec[index] = int(occupation_bin[i])
		index += 1

	timestamp_bin = bin(timestamp).strip('0b')
	for i in range(len(timestamp_bin)):
		vec[index] = int(timestamp_bin[i])
		index += 1

	return vec

def create_user_attribute_mapping():
	filename_read = '../data/ml-1m/users.dat'
	fp = open(filename_read, 'r')
	mapping = {}

	line = fp.readline()
	while line:
		temp = line.strip('\n').split('::')
		mapping[user_id] = []
		mapping[user_id].append(temp[1])
		mapping[user_id].append(int(temp[2]))
		mapping[user_id].append(int(temp[3]))

		line = fp.readline()


user_attribute_mapping = create_user_attribute_mapping()

filename_read = '../data/ml-1m/ratings_time.dat'
fp = open(filename_read, 'r')

line = fp.readline()
while line:
	temp = line.strip('\n').split('::')
	user_id = int(temp[0])
	movie_id = int(temp[1])
	true_rating = int(temp[3])
	timestamp = int(temp[4])
	line = fp.readline()

	gender = user_attribute_mapping[user_id][0]
	age = user_attribute_mapping[user_id][1]
	occupation = user_attribute_mapping[user_id][2]

	user_attribute = [gender, age, occupation, timestamp]
	context = create_context_vector(user_id, user_attribute)


