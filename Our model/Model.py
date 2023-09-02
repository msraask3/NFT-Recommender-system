import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from GraphGAT import GraphGAT  

class NFT_MARS(torch.nn.Module):
    def __init__(self, features, user_features, edge_index, batch_size, num_user, num_item, reg_parm, dim_x, DROPOUT, hop, path=None):
        super(NFT_MARS, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.reg_parm = reg_parm
        self.hop = hop

        self.edge_index = edge_index[:,:2]
        self.edge_index = torch.tensor(self.edge_index).t().contiguous().cuda()
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1,0]]), dim=1)
        
        v_feat, t_feat, p_feat, tr_feat = features
        self.v_feat = torch.tensor(v_feat, dtype=torch.float).cuda()
        self.t_feat = torch.tensor(t_feat, dtype=torch.float).cuda()
        self.p_feat = torch.tensor(p_feat, dtype=torch.float).cuda()
        self.tr_feat = torch.tensor(tr_feat, dtype=torch.float).cuda()

        self.user_features = torch.tensor(user_features, dtype=torch.float).cuda()

        self.v_gnn = GAT(self.v_feat, self.user_features, self.edge_index, batch_size, num_user, num_item, dim_x, DROPOUT, hop, dim_latent=None) 
        self.t_gnn = GAT(self.t_feat, self.user_features, self.edge_index, batch_size, num_user, num_item, dim_x, DROPOUT, hop, dim_latent=None) 
        self.p_gnn = GAT(self.p_feat, self.user_features, self.edge_index, batch_size, num_user, num_item, dim_x, DROPOUT, hop, dim_latent=None) 
        self.tr_gnn = GAT(self.tr_feat, self.user_features, self.edge_index, batch_size, num_user, num_item, dim_x, DROPOUT, hop, dim_latent=None) 

        self.id_embedding = nn.Embedding(num_user+num_item, dim_x)
        nn.init.xavier_normal_(self.id_embedding.weight)
        self.result_embed = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x))).cuda()

        # linear layers
        self.MLP_price = MLP_price(dim_in=dim_x*2*hop, dim_out=1)
        self.q_fc = nn.Linear(dim_x*hop, dim_x*hop, bias=False) # (d_model, d_model)
        self.k_fc = nn.Linear(dim_x*hop, dim_x*hop, bias=False) # (d_model, d_model)
        self.v_fc = nn.Linear(dim_x*hop, dim_x*hop, bias=False) # (d_model, d_model)


    def forward(self, user_nodes, pos_items, neg_items): # torch.Size([2048])   
          
        v_rep = self.v_gnn(self.id_embedding) 
        t_rep = self.t_gnn(self.id_embedding) 
        p_rep = self.p_gnn(self.id_embedding) 
        tr_rep = self.tr_gnn(self.id_embedding) # torch.Size([num_user+num_item, dim_x])
        self.v_representation = v_rep
        self.t_representation = t_rep
        self.p_representation = p_rep
        self.tr_representation = tr_rep

        representation = (v_rep + t_rep + p_rep + tr_rep)/4 # torch.Size([num_user+num_item, dim_x]) # fusion
        self.result_embed = representation
        user_tensor = representation[user_nodes]
        pos_tensor = representation[pos_items]
        neg_tensor = representation[neg_items]

        # QUERY
        Q = user_tensor # torch.Size([batch_size, dim_x])
        # KEY, VALUE
        pos_tensor_v = v_rep[pos_items]
        pos_tensor_t = t_rep[pos_items]
        pos_tensor_p = p_rep[pos_items]
        pos_tensor_tr = tr_rep[pos_items] # torch.Size([batch_size, dim_x]) 
        # create a matrix where each row is the average of pos_tensor_v, pos_tensor_t, pos_tensor_p, pos_tensor_tr
        K = torch.stack([pos_tensor_v.mean(dim=0), pos_tensor_t.mean(dim=0), pos_tensor_p.mean(dim=0), pos_tensor_tr.mean(dim=0)], dim=0)
        V = K.clone() # torch.Size([4, dim_x])

        # transform
        Q = self.q_fc(Q) # torch.Size([batch_size, dim_x])
        K = self.k_fc(K) # torch.Size([4, dim_x])
        V = self.v_fc(V)

        # calculate attention
        attention = torch.matmul(Q, K.transpose(1, 0)) / np.sqrt(K.shape[1]) # torch.Size([batch_size, 4])
        attention = F.softmax(attention, dim=1)
        attention = torch.matmul(attention, V) # torch.Size([batch_size, dim_x])
        user_tensor = attention
        
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

    def attention_score(self, dataset):
        attention_score = {}
        for data in dataset:

            user = data[0]
            pos_items = data[1]

            v_rep = self.v_representation
            t_rep = self.t_representation
            p_rep = self.p_representation 
            tr_rep = self.tr_representation 

            Q = self.result_embed[user]

            pos_tensor_v = v_rep[pos_items]
            pos_tensor_t = t_rep[pos_items]
            pos_tensor_p = p_rep[pos_items]
            pos_tensor_tr = tr_rep[pos_items]

            K = torch.stack([pos_tensor_v, pos_tensor_t, pos_tensor_p, pos_tensor_tr], dim=0)
            V = K.clone()

            Q = self.q_fc(Q) # torch.Size([2048, 512])
            K = self.k_fc(K) # torch.Size([4, 512])
            V = self.v_fc(V) # torch.Size([4, 512])

            attention = torch.matmul(Q, K.transpose(1, 0)) / np.sqrt(K.shape[1])
            attention_score[user] = attention.tolist()

        return attention_score


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

