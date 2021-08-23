import numpy as np
import torch
import torch.nn as nn

from autil import alignment
from autil.sparse_tensor import SpecialSpmm


class align_gcn(nn.Module):
    def __init__(self, kg_E, config):
        super(align_gcn, self).__init__()
        self.myprint = config.myprint
        self.config = config
        # Super Parameter
        self.relu = nn.ReLU(inplace=True)
        self.special_spmm = SpecialSpmm()  # sparse matrix multiplication
        self.sigmoid = nn.Sigmoid()

        self.kg_E = kg_E
        self.e_dim = config.rel_dim # dimension

        # GCN+highway
        self.gcnW1 = nn.Parameter(torch.zeros(size=(self.e_dim, self.e_dim)))
        self.highwayWr = nn.Parameter(torch.zeros(size=(self.e_dim, self.e_dim)))
        self.highwaybr = nn.Parameter(torch.zeros(size=(self.e_dim, 1)))


    def data_init(self, ent_newid_dict, ent_neigh_dict):
        ent_newid_array = np.array(list(ent_newid_dict.items()))
        self.ent_newid_tensor = torch.LongTensor(ent_newid_array[:,1])  # oldid list
        self.set_ent_adj(ent_neigh_dict, ent_newid_array, self.config.batch_size)
        if self.config.is_cuda:
            self.ent_newid_tensor = self.ent_newid_tensor.cuda()


    def set_ent_adj(self, ent_neigh_dict, ent_newid_array, batch_size):
        ''' Get neighbor relationships between entities '''
        new_ent_list = alignment.divide_array(ent_newid_array, batch_size)
        self.batch_num = 0
        self.kg_adj_list = []
        self.ent_oldids_list = []
        for batch_newid_array in new_ent_list:  # ent_list(30720) newid:(oldid, rid)
            oldid_list = batch_newid_array[:, 1].tolist()  # oldid
            self.ent_oldids_list.append(oldid_list)
            self.batch_num += 1

            # 邻居矩阵
            adj_row = []
            adj_col = []
            adj_data = []
            row_id = 0
            for h in batch_newid_array[:, 0]: # newid
                tr_list = ent_neigh_dict[h]
                t_list = [t for (t, r) in tr_list if h!=t ]
                du = len(t_list) + 1
                for t in t_list:
                    adj_row.append(row_id)  # h id => Line number
                    adj_col.append(t)
                    adj_data.append(1 / du)
                    assert t <= self.kg_E
                row_id += 1

            assert row_id == batch_newid_array.shape[0]

            adj_index = np.vstack((adj_row, adj_col))  # (2,D)
            self.kg_adj_list.append([adj_index, adj_data]) # eer_adj

        self.myprint('Divide the number of batches (GCN)：' + str(self.batch_num))


    def forward(self, right_embed):
        # GCN
        gcn_e_1 = self.add_diag_layer(right_embed)  # (E_new,dim)
        left_embed = right_embed[self.ent_newid_tensor, :]
        # highway
        gcn_e_2 = self.highway(left_embed, gcn_e_1)
        # gcn_e_3 = self.add_diag_layer(gcn_e_2)
        # output_layer = self.highway(gcn_e_2, gcn_e_3)  # (E_new,dim)
        return gcn_e_2  # (E_new,E)


    def add_diag_layer(self, right_embed):
        # batch
        embed_list = []
        for index in range(self.batch_num):
            batch_oldids_tensor, batch_adj_index, batch_adj_data = self.get_batch_adj(index)
            batch_size = batch_oldids_tensor.shape[0]
            embed_batch = self.add_diag_layer_batch(right_embed, batch_adj_index, batch_adj_data, batch_size)
            embed_list.append(embed_batch)
        all_embed = torch.cat(embed_list, dim=0)
        return all_embed


    def get_batch_adj(self, index):
        batch_oldids_tensor = torch.LongTensor(self.ent_oldids_list[index])
        batch_adj_index = torch.LongTensor(self.kg_adj_list[index][0])  # adj_index
        batch_adj_data = torch.FloatTensor(self.kg_adj_list[index][1])  # eer_adj_data
        if self.config.is_cuda:
            batch_oldids_tensor = batch_oldids_tensor.cuda()
            batch_adj_index = batch_adj_index.cuda()
            batch_adj_data = batch_adj_data.cuda()

        return batch_oldids_tensor, batch_adj_index, batch_adj_data

    # add a gcn layer
    def add_diag_layer_batch(self, e_inlayer, batch_adj_index, batch_adj_data, batch_size):
        #e_inlayer = self.dropout(e_inlayer)
        e_inlayer = torch.mm(e_inlayer, self.gcnW1)  # (E,dim)*(dim,dim) =>(E,dim)
        # e_adj  (batch_size,E)* (E,dim) =>(batch_size,dim)
        e_out = self.special_spmm(batch_adj_index, batch_adj_data, torch.Size([batch_size, self.kg_E]), e_inlayer)
        return self.relu(e_out)

    # add a highway layer
    def highway(self, e_layer1, e_layer2):
        # (E,dim) * (dim,dim)
        transform_gate = torch.mm(e_layer1, self.highwayWr) + self.highwaybr.squeeze(1)
        transform_gate = self.sigmoid(transform_gate)
        e_layer = transform_gate * e_layer2 + (1.0 - transform_gate) * e_layer1

        return e_layer

