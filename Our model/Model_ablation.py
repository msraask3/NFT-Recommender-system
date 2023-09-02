import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from GraphGAT import GraphGAT  

class MO(torch.nn.Module):
    def __init__(self, features, user_features, edge_index, batch_size, num_user, num_item, reg_parm, dim_x, DROPOUT, hop, what_feature='v', path=None):
        super(MO, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.reg_parm = reg_parm
        self.hop = hop

        self.edge_index = edge_index[:,:2]
        self.edge_index = torch.tensor(self.edge_index).t().contiguous().cuda()
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1,0]]), dim=1)
        self.what_feature = what_feature
        
        self.user_features = torch.tensor(user_features, dtype=torch.float).cuda()
        
        v_feat, t_feat, p_feat, tr_feat = features
        if what_feature == 'v':  
            self.v_feat = torch.tensor(v_feat, dtype=torch.float).cuda()
            self.v_gnn = GAT(self.v_feat, self.user_features, self.edge_index, batch_size, num_user, num_item, dim_x, DROPOUT, hop, dim_latent=None) 
        elif what_feature == 't':
            self.t_feat = torch.tensor(t_feat, dtype=torch.float).cuda()
            self.t_gnn = GAT(self.t_feat, self.user_features, self.edge_index, batch_size, num_user, num_item, dim_x, DROPOUT, hop, dim_latent=None) 
        elif what_feature == 'p':   
            self.p_feat = torch.tensor(p_feat, dtype=torch.float).cuda()
            self.p_gnn = GAT(self.p_feat, self.user_features, self.edge_index, batch_size, num_user, num_item, dim_x, DROPOUT, hop, dim_latent=None) 
        elif what_feature == 'tr':
            self.tr_feat = torch.tensor(tr_feat, dtype=torch.float).cuda()
            self.tr_gnn = GAT(self.tr_feat, self.user_features, self.edge_index, batch_size, num_user, num_item, dim_x, DROPOUT, hop, dim_latent=None) 
        elif what_feature == 'all':
            self.v_feat = torch.tensor(v_feat, dtype=torch.float).cuda()
            self.t_feat = torch.tensor(t_feat, dtype=torch.float).cuda()
            self.p_feat = torch.tensor(p_feat, dtype=torch.float).cuda()
            self.tr_feat = torch.tensor(tr_feat, dtype=torch.float).cuda()
            self.total_feat = torch.cat((self.v_feat, self.t_feat, self.p_feat, self.tr_feat), dim=1)
            self.gnn = GAT(self.total_feat, self.user_features, self.edge_index, batch_size, num_user, num_item, dim_x, DROPOUT, hop, dim_latent=None) 

        self.id_embedding = nn.Embedding(num_user+num_item, dim_x)
        nn.init.xavier_normal_(self.id_embedding.weight)
        self.result_embed = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x))).cuda()

        # linear layers
        self.MLP_price = MLP_price(dim_in=dim_x*2*hop, dim_out=1)
        self.q_fc = nn.Linear(dim_x*hop, dim_x*hop, bias=False) # (d_model, d_model)
        self.k_fc = nn.Linear(dim_x*hop, dim_x*hop, bias=False) # (d_model, d_model)
        self.v_fc = nn.Linear(dim_x*hop, dim_x*hop, bias=False) # (d_model, d_model)


    def forward(self, user_nodes, pos_items, neg_items): # torch.Size([2048])   
        
        if self.what_feature == 'v':
            v_rep = self.v_gnn(self.id_embedding) 
            self.representation = v_rep
        elif self.what_feature == 't':
            t_rep = self.t_gnn(self.id_embedding) 
            self.representation = t_rep
        elif self.what_feature == 'p':
            p_rep = self.p_gnn(self.id_embedding) 
            self.representation = p_rep
        elif self.what_feature == 'tr':
            tr_rep = self.tr_gnn(self.id_embedding) # torch.Size([num_user+num_item, dim_x])
            self.representation = tr_rep
        elif self.what_feature == 'all':
            total_rep = self.gnn(self.id_embedding) 
            self.representation = total_rep
    
        representation = self.representation # torch.Size([num_user+num_item, dim_x]) # fusion
        self.result_embed = representation
        user_tensor = representation[user_nodes]
        pos_tensor = representation[pos_items]
        neg_tensor = representation[neg_items]
        
        # 1) BPR pred
        pos_scores = torch.sum(user_tensor * pos_tensor, dim=1) # torch.Size([batch_size])
        neg_scores = torch.sum(user_tensor * neg_tensor, dim=1)
        
        # 2) Price pred
        user_pos_tensor = torch.cat((user_tensor, pos_tensor), dim=1) # torch.Size([batch_size, dim_x+dim_x]) 
        pred_price = self.MLP_price(user_pos_tensor) # torch.Size([batch_size, 1]) 
        
        return pos_scores, neg_scores, representation, pred_price

    def loss(self, data): 
        users, pos_items, neg_items, labels = data # batch data
        pos_scores, neg_scores, representation, pred_price = self.forward(users.cuda(), pos_items.cuda(), neg_items.cuda())

        # 1) BPR loss
        loss_value = -torch.sum(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        reg = (torch.norm(representation[users])**2
                + torch.norm(representation[pos_items])**2
                + torch.norm(representation[neg_items])**2) / 2
        loss_reg = self.reg_parm*reg / self.batch_size
        loss_BPR = loss_value + loss_reg

        # 2) BCE loss
        loss_Price = F.binary_cross_entropy(torch.sigmoid(pred_price.float()), labels.unsqueeze(1).float().cuda())

        return loss_BPR, loss_Price


    def accuracy(self, dataset, indices, topk=10, neg_num=100):
        all_set = set(list(np.arange(neg_num))) 
        num_user = len(dataset)
        recall_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) # topK=10,20,30,40,50
        ndcg_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0])   # topK=10,20,30,40,50

        for data in dataset: # user loop
            user = data[0]
            pos_items = data[1:]
            neg_items = [x for x in indices if x not in pos_items] # popularity-based negative sampling
            neg_items = neg_items[:neg_num]
            neg_items = list(neg_items)

            batch_user_tensor = torch.tensor(user).cuda() 
            batch_pos_tensor = torch.tensor(pos_items).cuda()
            batch_neg_tensor = torch.tensor(neg_items).cuda()

            user_embed = self.result_embed[batch_user_tensor]
            pos_v_embed = self.result_embed[batch_pos_tensor]
            neg_v_embed = self.result_embed[batch_neg_tensor]

            num_pos = len(pos_items)
            pos_score = torch.sum(pos_v_embed*user_embed, dim=1)
            neg_score = torch.sum(neg_v_embed*user_embed, dim=1)

            ###################################### Select topK based on scores ######################################
            for k, topk in enumerate(range(10, 51, 10)):
                _, index_of_rank_list = torch.topk(torch.cat((neg_score, pos_score)), topk) # indices of top5 items (e.g., [0, 10, 101, 130, 315])
                # Recall
                index_set = set([iofr.cpu().item() for iofr in index_of_rank_list])
                num_hit = len(index_set.difference(all_set))                                # only the indices>=100 are included in the pos_items (e.g., [130, 315]])
                recall = float(num_hit/num_pos)
                # NDCG
                ndcg_score = 0.0
                for i in range(num_pos):
                    label_pos = neg_num + i
                    if label_pos in index_of_rank_list:
                        index = list(index_of_rank_list.cpu().numpy()).index(label_pos)
                        ndcg_score = ndcg_score + math.log(2) / math.log(index + 2)
                ndcg = ndcg_score/num_pos
                # append to the list
                recall_list[k] += recall
                ndcg_list[k] += ndcg

        return recall_list/num_user, ndcg_list/num_user


class GAT(torch.nn.Module):
    def __init__(self, features, user_features, edge_index, batch_size, num_user, num_item, dim_id, DROPOUT, hop, dim_latent=None):
        super(GAT, self).__init__()
        self.features = features
        self.user_features = user_features
        self.edge_index = edge_index
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.DROPOUT = DROPOUT
        self.dim_latent = dim_latent
        self.hop = hop

        self.dim_feat = features.size(1)
        self.user_dim_feat = user_features.size(1)

        # self.preference = nn.Embedding(num_user, self.dim_latent)
        # nn.init.xavier_normal_(self.preference.weight).cuda()

        # self.MLP = nn.Linear(self.dim_feat, self.dim_latent) 
        self.user_MLP = nn.Linear(self.user_dim_feat, self.dim_feat) # to match the dimensionality of the item features

        if self.dim_latent:
            self.conv_embed_1 = GraphGAT(self.dim_latent, self.dim_latent, self.DROPOUT, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight) 
        else:
            self.conv_embed_1 = GraphGAT(self.dim_feat, self.dim_feat, self.DROPOUT, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight)

        if hop == 2:
            self.conv_embed_2 = GraphGAT(self.dim_id, self.dim_id, self.DROPOUT, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_2.weight)
            self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer2.weight)
            self.g_layer2 = nn.Linear(self.dim_id, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer2.weight)
        if hop == 3:
            self.conv_embed_2 = GraphGAT(self.dim_id, self.dim_id, self.DROPOUT, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_2.weight)
            self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer2.weight)
            self.g_layer2 = nn.Linear(self.dim_id, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer2.weight)
            self.conv_embed_3 = GraphGAT(self.dim_id, self.dim_id, self.DROPOUT, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_3.weight)
            self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer3.weight)
            self.g_layer3 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.g_layer3.weight)
        if hop == 4:
            self.conv_embed_2 = GraphGAT(self.dim_id, self.dim_id, self.DROPOUT, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_2.weight)
            self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer2.weight)
            self.g_layer2 = nn.Linear(self.dim_id, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer2.weight)
            self.conv_embed_3 = GraphGAT(self.dim_id, self.dim_id, self.DROPOUT, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_3.weight)
            self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer3.weight)
            self.g_layer3 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.g_layer3.weight)
            self.conv_embed_4 = GraphGAT(self.dim_id, self.dim_id, self.DROPOUT, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_4.weight)
            self.linear_layer4 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer4.weight)
            self.g_layer4 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.g_layer4.weight)
        if hop == 5:
            self.conv_embed_2 = GraphGAT(self.dim_id, self.dim_id, self.DROPOUT, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_2.weight)
            self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer2.weight)
            self.g_layer2 = nn.Linear(self.dim_id, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer2.weight)
            self.conv_embed_3 = GraphGAT(self.dim_id, self.dim_id, self.DROPOUT, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_3.weight)
            self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer3.weight)
            self.g_layer3 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.g_layer3.weight)
            self.conv_embed_4 = GraphGAT(self.dim_id, self.dim_id, self.DROPOUT, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_4.weight)
            self.linear_layer4 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer4.weight)
            self.g_layer4 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.g_layer4.weight)
            self.conv_embed_5 = GraphGAT(self.dim_id, self.dim_id, self.DROPOUT, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_5.weight)
            self.linear_layer5 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer5.weight)
            self.g_layer5 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.g_layer5.weight)


    def forward(self, id_embedding):
        hop = self.hop

        # item_features = torch.tanh(self.MLP(self.features)) if self.dim_latent else self.features   # dim_feat -> dim_latent
        # user_features = self.preference.weight                                                      # dim_latent

        item_features = nn.Embedding.from_pretrained(self.features, freeze=True).weight           # dim_feat
        user_features = torch.tanh(self.user_MLP(self.user_features))                             # user_dim_feat -> dim_feat

        x = torch.cat((item_features, user_features), dim=0) 
        x = F.normalize(x).cuda()

        if hop == 1:
            h = F.leaky_relu(self.conv_embed_1(x, self.edge_index, None)) 
            x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding.weight
            x = F.leaky_relu(self.g_layer1(h)+x_hat)
            return x
        
        elif hop == 2:
            h = F.leaky_relu(self.conv_embed_1(x, self.edge_index, None)) 
            x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding.weight
            x_1 = F.leaky_relu(self.g_layer1(h)+x_hat)
            h = F.leaky_relu(self.conv_embed_2(x_1, self.edge_index, None))
            x_hat = F.leaky_relu(self.linear_layer2(x_1)) + id_embedding.weight
            x_2 = F.leaky_relu(self.g_layer2(h)+x_hat)
            x = torch.cat((x_1, x_2), dim=1)
            return x 
        
        elif hop == 3:
            h = F.leaky_relu(self.conv_embed_1(x, self.edge_index, None))
            x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding.weight
            x_1 = F.leaky_relu(self.g_layer1(h)+x_hat)
            h = F.leaky_relu(self.conv_embed_2(x_1, self.edge_index, None))
            x_hat = F.leaky_relu(self.linear_layer2(x_1)) + id_embedding.weight
            x_2 = F.leaky_relu(self.g_layer2(h)+x_hat)
            h = F.leaky_relu(self.conv_embed_3(x_2, self.edge_index, None))
            x_hat = F.leaky_relu(self.linear_layer3(x_2)) + id_embedding.weight
            x_3 = F.leaky_relu(self.g_layer3(h)+x_hat)
            x = torch.cat((x_1, x_2, x_3), dim=1)
            return x
        
        elif hop == 4:
            h = F.leaky_relu(self.conv_embed_1(x, self.edge_index, None))
            x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding.weight
            x_1 = F.leaky_relu(self.g_layer1(h)+x_hat)
            h = F.leaky_relu(self.conv_embed_2(x_1, self.edge_index, None))
            x_hat = F.leaky_relu(self.linear_layer2(x_1)) + id_embedding.weight
            x_2 = F.leaky_relu(self.g_layer2(h)+x_hat)
            h = F.leaky_relu(self.conv_embed_3(x_2, self.edge_index, None))
            x_hat = F.leaky_relu(self.linear_layer3(x_2)) + id_embedding.weight
            x_3 = F.leaky_relu(self.g_layer3(h)+x_hat)
            h = F.leaky_relu(self.conv_embed_4(x_3, self.edge_index, None))
            x_hat = F.leaky_relu(self.linear_layer4(x_3)) + id_embedding.weight
            x_4 = F.leaky_relu(self.g_layer4(h)+x_hat)
            x = torch.cat((x_1, x_2, x_3, x_4), dim=1)
            return x
        
        elif hop == 5:
            h = F.leaky_relu(self.conv_embed_1(x, self.edge_index, None))
            x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding.weight
            x_1 = F.leaky_relu(self.g_layer1(h)+x_hat)
            h = F.leaky_relu(self.conv_embed_2(x_1, self.edge_index, None))
            x_hat = F.leaky_relu(self.linear_layer2(x_1)) + id_embedding.weight
            x_2 = F.leaky_relu(self.g_layer2(h)+x_hat)
            h = F.leaky_relu(self.conv_embed_3(x_2, self.edge_index, None))
            x_hat = F.leaky_relu(self.linear_layer3(x_2)) + id_embedding.weight
            x_3 = F.leaky_relu(self.g_layer3(h)+x_hat)
            h = F.leaky_relu(self.conv_embed_4(x_3, self.edge_index, None))
            x_hat = F.leaky_relu(self.linear_layer4(x_3)) + id_embedding.weight
            x_4 = F.leaky_relu(self.g_layer4(h)+x_hat)
            h = F.leaky_relu(self.conv_embed_5(x_4, self.edge_index, None))
            x_hat = F.leaky_relu(self.linear_layer5(x_4)) + id_embedding.weight
            x_5 = F.leaky_relu(self.g_layer5(h)+x_hat)
            x = torch.cat((x_1, x_2, x_3, x_4, x_5), dim=1)
            return x
    

# Price prediction layer
class MLP_price(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP_price, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear_layer1 = nn.Linear(self.dim_in, self.dim_in//2)
        nn.init.xavier_normal_(self.linear_layer1.weight)
        self.linear_layer2 = nn.Linear(self.dim_in//2, self.dim_out)
        nn.init.xavier_normal_(self.linear_layer2.weight)

    def forward(self, x):
        x = F.leaky_relu(self.linear_layer1(x))
        x = self.linear_layer2(x)

        return x

