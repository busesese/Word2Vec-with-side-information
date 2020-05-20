# author: WenYi
# email: 1244058349@qq.com

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SkigGram(nn.Module):
	def __init__(self, embedding_size, embedding_dim, side_information):
		"""
		parameter init
		param: embedding_size, int, unique word number
		param: embedding_dim, int embedding dimension
		param: side_information, dict parameter save dict like {'item_attribute_side_1': unique_count, 'item_attribute_side_2': unique_count, 
		'item_attribute_side_3': unique_count, 'item_attribute_side_4': unique_count} note:each key in side_information dict must include '_side_x' part
		"""
		super(SkigGram, self).__init__()
		self.side_information = side_information
		self.embedding_dim = embedding_dim

		# center word embedding
		self.center_word_embed = nn.Embedding(embedding_size, embedding_dim)
		# neighbor word embedding
		self.neighbor_word_embed = nn.Embedding(embedding_size, embedding_dim)

		# side information embedding
		for key, val in side_information.items():
			setattr(self, key + '_embed', nn.Embedding(val, embedding_dim))

		# init side information weight 
		self.embedding_weight = torch.rand((len(side_information) + 1, 1), requires_grad=True)

		# embedding weight init
		# self._weight_init()

	def _weight_init(self):
		nn.init.uniform_(self.center_word_embed.weight.data, -1.0/self.embedding_dim, 1.0/self.embedding_dim)
		nn.init.constant_(self.neighbor_word_embed.weight.data, 0)
		for key, _ in self.side_information.items():
			nn.init.uniform_(getattr(self, key + '_embed').weight.data, -1.0/self.embedding_dim, 1.0/self.embedding_dim)

	def forward(self, center_word, neighor_word, neg_word):
		"""
		forward network
		param: center_word, Tensor,  [batch_size, len(side_information) + 1]  note: len(side_information) + 1 is the lenght of center word and word side information 
		param: neighor_word, Tensor, [batch_size, 1]
		param: neg_word, Tensor, [batch_size, len(neg_word)]
		"""
		information_list = []
		# center word
		if center_word.size()[1] == 1:
			embed_center_word = self.center_word_embed(center_word) # batch_size * 1 * embed_dim
			embed_center_word = embed_center_word.squeeze() # batch_size * embed_dim
		else:
			embed_center_word = self.center_word_embed(center_word[:, 0]) # batch_size * embed_dim
		information_list.append(embed_center_word)
		# neighbor word
		embed_neighbor_word = self.neighbor_word_embed(neighor_word) # batch_size * 1 * embed_dim
		# neg word
		embed_neg_word = self.neighbor_word_embed(neg_word) # batch_size * embed_dim

		# side information
		if self.side_information:
			for key ,val in self.side_information.items():
				for i in range(1, center_word.size()[1] + 1):
					if str(i) in key:
						information_list.append(getattr(self, key + '_embed')(center_word[:, i]))

		# word and side information embeding list
		information_embed = torch.cat(information_list, dim=0).view(len(information_list), -1, self.embedding_dim)
		weight_sum_pooling = information_embed * self.embedding_weight.view(-1, 1, 1)
		embed_center_word_side_information = torch.sum(weight_sum_pooling, dim=0)

		score = torch.sum(torch.mul(embed_center_word_side_information, embed_neighbor_word.squeeze()), dim=1)
		score = torch.clamp(score, max=10, min=-10)
		score = -F.logsigmoid(score)

		neg_score = torch.bmm(embed_neg_word, embed_center_word_side_information.unsqueeze(2)).squeeze()
		neg_score = torch.clamp(neg_score, max=10, min=-10)
		neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)
		return torch.mean(score + neg_score)

	def save_embedding(self, id2word, file_name):
		embedding = self.u_embeddings.weight.cpu().data.numpy()
		with open(file_name, 'w') as f:
			f.write('%d %d\n' % (len(id2word), self.emb_dimension))
			for wid, w in id2word.items():
				e = ' '.join(map(lambda x: str(x), embedding[wid]))
				f.write('%s %s\n' % (w, e))


if __name__ == "__main__":

	parameter_dict = {'information_side_1': 20, 'information_side_2': 10}
	# parameter_dict = dict()
	embedding_size = 60
	embedding_dim = 16

	skp = SkigGram(embedding_size, embedding_dim, parameter_dict)

	center_word = torch.from_numpy(np.array([[10, 3, 6], [1, 14, 3], [10, 2, 7], [3, 5, 8]]))
	neighbor_word = torch.from_numpy(np.array([[15], [21], [9], [5]]))
	neg_word = torch.from_numpy(np.array([[11, 18, 20], [17, 14, 4], [5,9,6], [12,32,14]]))
	score = skp(center_word, neighbor_word, neg_word)
	print(score)









