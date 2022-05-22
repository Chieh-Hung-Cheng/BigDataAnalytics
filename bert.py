import numpy as np
import pandas as pd
import os
import sys

import tensorflow
from keras_bert import load_vocabulary
from keras_bert import Tokenizer
from keras_bert import load_trained_model_from_checkpoint

import doc_utils


  	# 提取embedding的CLS部分，存入空array

def bert_test():
	shoes = pd.read_pickle(os.path.join(doc_utils.raw_pth, 'Shoes05211728.pkl'))
	text = shoes['SalePageTitle'].tolist()

	pretrained = os.path.join(doc_utils.data_pth, 'bert')
	dictpath = os.path.join(pretrained, 'vocab.txt')
	config_path = os.path.join(pretrained, 'bert_config.json')
	checkpoint_path = os.path.join(pretrained, 'bert_model.ckpt')
	print("The pretrained model is loaded")
	# 載入中文預訓練好的BERT語料庫

	model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
	bert_token_dict = load_vocabulary(dictpath)
	bert_tokenizer = Tokenizer(bert_token_dict)

	data = np.zeros((9594, 768))  # 建立空的nparray
	for i in range(0, 9594):
		tokens = text[i]
		indices, segments = bert_tokenizer.encode(first=text[i])
		# 轉化成向量(向量維度768是預設輸出長度)

		predicts = model.predict(np.array([indices]), np.array([segments]))[0]
		embedding = predicts[0]
		data[i, :] = embedding


if __name__ == '__main__':
	bert_test()