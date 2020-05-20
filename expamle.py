from utils import getNeagtives, Word2VecDataSet
from word2vecsf import Word2VecSF
import numpy as np 
import pandas as pd 
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "5, 6, 7"


center_word = np.random.randint(1, 100, size=(10000, 3))
neighbor_word = np.random.randint(1, 100, size=10000)
neg_word = np.random.randint(1, 100, size=(10000, 2))
data = (center_word, neighbor_word, neg_word)

embedding_siz = 101
side_inforamtion = {'center_side_1': 101, 'center_side_2': 101}
gpu_number = 0

w2v = Word2VecSF(data, embedding_siz, side_inforamtion, gpu_number)
w2v.train()
print(w2v.skipgram.neighbor_word_embed.weight)
