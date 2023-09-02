import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import matplotlib.pyplot as plt
import random
from torchcontrib.optim import SWA
import wandb

from DataLoad import DataLoad
from Model_MGAT import MGAT

class Net:
    def __init__(self, args):

        np.random.seed(args.seed)
        random.seed(args.seed)

        self.model_name = args.model_name
        self.collection = args.collection
        self.data_path = args.data_path
        self.PATH_weight_load = args.PATH_weight_load
        self.PATH_weight_save = args.PATH_weight_save
        self.l_r = args.l_r
        self.weight_decay = args.weight_decay
        self.batch_size = args.batch_size
        self.dim_x = args.dim_x
        self.num_epoch = args.num_epoch
        self.num_workers = args.num_workers
        self.reg_parm = args.reg_parm
        self.neg_sample = args.neg_sample
        self.loss_alpha = args.loss_alpha
        self.number = args.number
        self.attention_dropout = args.attention_dropout
        self.SWA = args.SWA
        self.device = "cuda:0"
        self.patience = args.num_epoch # for early stopping
        self.hop = args.hop

        # data_path: 'dataset/collections/bayc'
        self.data_path = os.path.join(self.data_path, self.collection)
        os.makedirs(self.data_path, exist_ok=True)

        # save_path: 'saved/MGAT/bayc'
        self.save_path = os.path.join(self.PATH_weight_save, self.model_name, self.collection)
        os.makedirs(self.save_path, exist_ok=True)

        ###################################### Load data ######################################
        print('Loading data  ...')
        num_user_item = np.load(os.path.join(self.data_path, 'num_user_item.npy'), allow_pickle=True).item()
        self.num_item = num_user_item['num_item']
        self.num_user = num_user_item['num_user']
        print(f"num_user: {self.num_user}, num_item: {self.num_item}")

        self.train_dataset = DataLoad(self.data_path, self.num_user, self.num_item, 0)
        for i in range(1, self.neg_sample+1): 
            self.train_dataset += DataLoad(self.data_path, self.num_user, self.num_item, i)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers)

        self.edge_index = np.load(os.path.join(self.data_path, 'train.npy'))
        self.val_dataset = np.load(os.path.join(self.data_path, 'val.npy'), allow_pickle=True)
        self.test_dataset = np.load(os.path.join(self.data_path, 'test.npy'), allow_pickle=True)

        self.v_feat = np.load(os.path.join(self.data_path, 'image_feat.npy'))
        self.t_feat = np.load(os.path.join(self.data_path, 'text_feat.npy'))
        self.p_feat = np.load(os.path.join(self.data_path, 'price_feat.npy'))
        self.tr_feat = np.load(os.path.join(self.data_path, 'transaction_feat.npy'))

        self.user_feat = np.load(os.path.join(self.data_path, 'user_feat.npy'))

        self.indices_valid = np.load(os.path.join(self.data_path, 'indices_valid.npy'), allow_pickle=True)
        self.indices_test = np.load(os.path.join(self.data_path, 'indices_test.npy'), allow_pickle=True)

        ###################################### Load model ######################################
        print('Loading model  ...')
        if self.model_name == 'MGAT':
            self.features = [self.v_feat, self.t_feat, self.p_feat, self.tr_feat]
            self.model = MGAT(self.features, self.user_feat, self.edge_index, self.batch_size, self.num_user, self.num_item,
                               self.reg_parm, self.dim_x, self.attention_dropout, self.hop, self.data_path).cuda()

        # Optimizer
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.l_r}],
                                            weight_decay=self.weight_decay)
        if self.SWA == True:
            self.optimizer = SWA(self.optimizer, swa_start=5, swa_freq=3, swa_lr=0.001)
            self.optimizer.defaults=self.optimizer.optimizer.defaults
 
        # Load model
        if self.PATH_weight_load and os.path.exists(self.PATH_weight_load):
            self.model.load_state_dict(torch.load(self.PATH_weight_load))
            print('module weights loaded....')


    def run(self):

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        best_recall, best_ndcg = 0.0, 0.0

        print(f'{self.collection}: Start training ...')
        patience_count = 0
        for epoch in range(self.num_epoch):

            ###################################### Train ######################################
            self.model.train()
            loss_batch, loss_BPR_batch, loss_Price_batch = 0.0, 0.0, 0.0
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                loss_BPR, loss_Price = self.model.loss(data)
                loss = (1-self.loss_alpha)*loss_BPR + self.loss_alpha*loss_Price
                loss.backward()
                self.optimizer.step()

                loss_batch += loss
                loss_BPR_batch += loss_BPR
                loss_Price_batch += loss_Price

            if self.SWA == True:
                self.optimizer.update_swa()

            loss_avg = loss_batch.item() / self.batch_size
            loss_BPR_avg = loss_BPR_batch.item() / self.batch_size
            loss_Price_avg = loss_Price_batch.item() / self.batch_size

            ###################################### Evaluate ######################################
            self.model.eval()
            with torch.no_grad():
                # Valid
                total_recall, total_ndcg = self.model.accuracy(self.val_dataset, self.indices_valid)
                recall, ndcg = total_recall[-1], total_ndcg[-1] # top50 only
                # Test
                total_recall_test, total_ndcg_test = self.model.accuracy(self.test_dataset, self.indices_test)
                recall_test, ndcg_test = total_recall_test[-1], total_ndcg_test[-1] # top50 only

            # Save best model
            if recall > best_recall: 
                best_recall = recall
                path = os.path.join(self.save_path, f'{self.number}_model.pt')
                torch.save(self.model.state_dict(), path)
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    break
            
            ###################################### save Epoch ######################################
            print(
                '{0}-th Loss:{1:.4f}, BPR:{2:.4f} Price: {3:.4f} / Recall:{4:.4f} NDCG:{5:.4f} / Best Recall:{6:.4f}'.format(
                    epoch, loss_avg, loss_BPR_avg, loss_Price_avg, 
                    recall, ndcg, 
                    best_recall
                    ))

            epoch = {'loss':loss_avg, 'loss_BPR': loss_BPR_avg, 'loss_Price': loss_Price_avg, 
                     'recall': recall, 'ndcg': ndcg, # top50
                     'recall_test': recall_test, 'ndcg_test': ndcg_test, # top50
                     'recall_top10': total_recall[0], 'ndcg_top10': total_ndcg[0],
                     'recall_top20': total_recall[1], 'ndcg_top20': total_ndcg[1],
                     'recall_top30': total_recall[2], 'ndcg_top30': total_ndcg[2],
                     'recall_top40': total_recall[3], 'ndcg_top40': total_ndcg[3],
                     'recall_top50': total_recall[4], 'ndcg_top50': total_ndcg[4],
                     'recall_test_top10': total_recall_test[0], 'ndcg_test_top10': total_ndcg_test[0],
                     'recall_test_top20': total_recall_test[1], 'ndcg_test_top20': total_ndcg_test[1],
                     'recall_test_top30': total_recall_test[2], 'ndcg_test_top30': total_ndcg_test[2],
                     'recall_test_top40': total_recall_test[3], 'ndcg_test_top40': total_ndcg_test[3],
                     'recall_test_top50': total_recall_test[4], 'ndcg_test_top50': total_ndcg_test[4],
                     }
            wandb.log(epoch)


if __name__ == '__main__':

    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID" # use GPU #0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--model_name', default='MGAT', help='Model name.')
    parser.add_argument('--collection', default='bayc', help='Collection name.')
    parser.add_argument('--data_path', default='dataset/collections', help='Dataset path')
    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', default='saved', help='Writing weight filename.')
    parser.add_argument('--l_r', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
    parser.add_argument('--dim_x', type=int, default=64, help='embedding dimension')
    parser.add_argument('--num_epoch', type=int, default=200, help='Epoch number')
    parser.add_argument('--num_workers', type=int, default=0, help='Workers number')
    parser.add_argument('--reg_parm', type=float, default=0.001, help='Regularizer')
    parser.add_argument('--neg_sample', type=int, default=5, help='num of negative samples for training')
    parser.add_argument('--loss_alpha', type=float, default=0, help='alpha for loss')
    parser.add_argument('--number', type=int, default=0, help='A number used to distinguish between runs.')
    parser.add_argument('--attention_dropout', type=float, default=0.2, help='dropout ratio')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed')
    parser.add_argument('--optimizer', default='Adam', help='Optimizer')
    parser.add_argument('--SWA', default=False, help='Optimizer')
    parser.add_argument('--hop', type=int, default=2, help='The number of hop')
    parser.add_argument('--project', default='project', help='the name of project')
    args = parser.parse_args()

    MGAT = Net(args)
    wandb.init(project="YOUR_PROJECT", name=f'{args.collection}_{args.loss_alpha}', entity="YOUR_ENTITY", config={k: v for k, v in args._get_kwargs()})
    MGAT.run()
    wandb.finish()