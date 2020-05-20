# author: WenYi
# email: 1244058349@qq.com

import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import Dataset


def negativeTable(table_size, word_frequency):
	vocab_size = len(word_frequency)
	word_frequency_count = list(word_frequency.values())
	word_frequency_count_sum = sum([count**0.75 for count in word_frequency_count])
	sample_ratio = [count**0.75/word_frequency_count_sum for count in word_frequency_count]
	idx = 0
	ratio = sample_ratio[idx]
	table = [0]*int(table_size)
	for i in range(int(table_size)):
		table[i] = idx
		if i / table_size > ratio:
			idx += 1
			ratio += sample_ratio[idx]
		if idx > vocab_size:
			idx = vocab_size - 1
	return table


def getNeagtives(word, table, neg_number=5):
	table_size = len(table)
	neg_result = list()

	while neg_number > 0:
		idx = np.random.randint(0, table_size)
		if table[idx] != word:
			neg_result.append(table[idx])
			neg_number -= 1
	return neg_result

class Word2VecDataSet(Dataset):
	def __init__(self, data):
		self.center_word = data[0]
		self.neighbor_word = data[1]
		self.neg_word = data[2]

	def __getitem__(self, index):
		center_word = self.center_word[index]
		neighbor_word = self.neighbor_word[index]
		neg_word = self.neg_word[index]
		data = (center_word, neighbor_word, neg_word)
		return data

	def __len__(self):
		return len(self.center_word)






