import random
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

# ONLY FOR TRAIN SET
class DataLoad(Dataset): 
	def __init__(self, path, num_user, num_item, neg_sample_num):
		super(DataLoad, self).__init__()
		self.data = np.load(os.path.join(path, 'train.npy'), allow_pickle=True)
		self.pop_indices = np.load(os.path.join(path, 'indices_valid.npy'), allow_pickle=True) 
		# self.adj_lists = np.load(os.path.join(path, 'final_adj_dict.npy')).item()
		# self.all_set = set(range(num_user, num_user+num_item))
		self.adj_lists = np.load(os.path.join(path, 'adj_dict.npy'), allow_pickle=True).item() 
		self.all_set = set(range(num_item))
		self.neg_sample_num = neg_sample_num

	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, index):
		user, pos_item, label = self.data[index]
		neg_item = [x for x in self.pop_indices if x not in self.adj_lists[user]][self.neg_sample_num] # popularity based negative sampling
		#neg_item = self.all_set.difference(self.adj_lists[user])
		return [user, pos_item, neg_item, label]

