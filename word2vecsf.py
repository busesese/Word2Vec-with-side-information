# author: WenYi
# email: 1244058349@qq.com

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import Word2VecDataSet
import torch.optim as optim
from SkipGram import SkigGram
import os
from data_paraller import BalancedDataParallel


class Word2VecSF:
	def __init__(self, data, embedding_size, side_inforamtion, gpu_number, embedding_dim=100, batch_size=64,
				 epochs=5, lr= 0.001):
		"""
		param: data, train data include (center_word, neighbor_word, neggtive_word) is ([[word1, side1, side2]], [word2], [[neg_word1, neg_word2]]), 
			   detail explanation read word2vec alogrithm, note: if include side information in center word
		param: embedding_size, word number of dataset
		param: side_information, dict, include each side information name and distinct number {'xx_side_1': number, 'xx_side_2': number} 
		       note: side information name must include 'side_x' like 'side_1', 'side_2'
		param: gpu_number, int, gpu number for train model
		param: embedding_dim, int embedding dimension,
		param: batch_size, int, data batch size,
		param: epochs, int, training epoch number
		param: lr, float, learning rate
		"""
		dataset = Word2VecDataSet(data)
		if gpu_number == 0:
			self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
		else:
			self.dataloader = DataLoader(dataset, batch_size=batch_size*gpu_number, shuffle=False, num_workers=4)
		self.embedding_size = embedding_size
		self.embedding_dim = embedding_dim
		self.batch_size = batch_size
		self.gpu_number = gpu_number
		self.epochs = epochs
		self.learning_rate = lr
		self.skipgram = SkigGram(self.embedding_size, self.embedding_dim, side_inforamtion)

		if gpu_number != 0:
			if torch.cuda.is_available() == False:
				raise ValueError("there is not gpu available please check you input parameter 'user_gpu' ")
			if gpu_number == 1:
				self.skipgram.cuda()
			else:
				gpu0_bsz = batch_size//2
				acc_grad = 1
				self.skipgram = BalancedDataParallel(gpu0_bsz//acc_grad, self.skipgram, dim=0).cuda()

	def train(self):
		for epoch in range(self.epochs):
			print("Epoch %d/%d -------" % (epoch+1, self.epochs))
			optimizer = optim.Adam(self.skipgram.parameters(), lr=self.learning_rate)
			running_loss = 0.0
			for i, (center_word, neighbor_word, neg_word) in enumerate(tqdm(self.dataloader)):
				if self.gpu_number != 0:
					center_word = center_word.cuda()
					neighbor_word = neighbor_word.cuda()
					neg_word = neg_word.cuda()
				optimizer.zero_grad()
				loss = self.skipgram(center_word, neighbor_word, neg_word)
				loss.mean().backward()
				optimizer.step()

				running_loss = running_loss * 0.9 + loss.item()*0.1
				if i > 0 and i % 500 == 0:
					print("Loss: " + str(running_loss))
			# self.skipgram.save_embedding(self.data.id2word, self.output_file_name)

	def word_embedding(self, word, word2index):
		"""
		get the word embedding
		param: word, string, origin word of dataset
		param: word2index, dict, origin word and id mapping dict,
		note: the word usually label encoder to index and model only can get the index embedding
		you should input the word2index mapping dict
		"""
		index = word2index[word]
		word_embedding = self.skipgram.center_word_embed.weight.detach().cpu().numpy()
		vector = word_embedding[index]
		return vector






