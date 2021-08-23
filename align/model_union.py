import numpy as np
import torch
import torch.nn as nn

from align import model_util
from align.model_gcn import align_gcn
from autil import alignment, fileUtil
from autil.sparse_tensor import SpecialSpmm


class align_union(nn.Module):
    def __init__(self, kgs_data, config):
        super(align_union, self).__init__()
        self.myprint = config.myprint
        self.config = config

        # Super Parameter
        self.leakyrelu = nn.LeakyReLU(config.LeakyReLU_alpha)  # LeakyReLU_alpha=0.2
        self.relu = nn.ReLU(inplace=True)
        self.special_spmm = SpecialSpmm()  # sparse matrix multiplication

        self.kg_E = kgs_data.kg_E
        self.kg_R = kgs_data.kg_R
        #self.kg_E_new = kgs_data.kg_E

        # Entity name embedding ##############
        self.e_dim = config.rel_dim
        self.r_dim = self.e_dim * 2
        self.kg_name_embed = kgs_data.ename_embed
        self.kg_name_w = nn.Parameter(torch.zeros(size=(300, self.e_dim)))
        self.kg_name_b = nn.Parameter(torch.zeros(size=(self.e_dim, 1)))

        # Set Relation neighbor parameters
        self.set_R_adj(kgs_data.rel_triple)

        # GCN Layer
        self.gcn_model = align_gcn(kgs_data.kg_E, config)
        if self.config.is_cuda:
            self.kg_name_embed = self.kg_name_embed.cuda()
            self.gcn_model = self.gcn_model.cuda()

        ########################
        # Relational embedding
        self.w_R_Left = nn.Parameter(torch.zeros(size=(self.e_dim, self.e_dim))) # W_r
        self.w_R_Right = nn.Parameter(torch.zeros(size=(self.e_dim, self.e_dim)))
        self.w_atten_r = nn.Parameter(torch.zeros(size=(self.r_dim, 1)))  # R attention参数

        ## Parameter initialization ##############
        for m in self.parameters():
            if isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.data, gain=1.414)  # Xavier正态分布
        ## Parameter list
        params = list(self.parameters()) + list(self.gcn_model.parameters())
        self.model_params = [{'params': params}]

        # Set training set, test set, candidate set and negative sample
        self.links_old_init(kgs_data)
        self.beg_gcn = self.is_regen = False


    def set_R_adj(self, rel_triple):
        # (R,E)  array[r,h]= sum(hr)
        r_head_array = np.zeros((self.kg_R, self.kg_E))
        r_tail_array = np.zeros((self.kg_R, self.kg_E))
        r_mat_row = []
        r_mat_col = []
        r_mat_data = []
        for (h, r, t) in rel_triple:
            r_head_array[r][h] += 1
            r_tail_array[r][t] += 1

            r_mat_row.append(h)
            r_mat_col.append(t)
            r_mat_data.append(r)

            r_mat_row.append(t)
            r_mat_col.append(h)
            r_mat_data.append(r)

        # r_head:array[r,h]=1,  r_tail:array[r,t]=1
        self.r_head = torch.FloatTensor(r_head_array)
        self.r_tail = torch.FloatTensor(r_tail_array)
        # Auxiliary information
        self.tensor_zero = torch.tensor(0.)
        r_head_sum = torch.unsqueeze(torch.sum(self.r_head, dim=-1), -1)  # (R,E)
        r_tail_sum = torch.unsqueeze(torch.sum(self.r_tail, dim=-1), -1)  # (R,E)
        self.r_head_sum = torch.where(r_head_sum == 0, torch.tensor(0.), 1. / r_head_sum)  # Instead of countdown
        self.r_tail_sum = torch.where(r_tail_sum == 0, torch.tensor(0.), 1. / r_tail_sum)  # Instead of countdown

        eer_adj_index = np.vstack((r_mat_row, r_mat_col))  # (2,D)
        self.eer_adj_index = torch.LongTensor(eer_adj_index)
        self.eer_adj_data = torch.LongTensor(r_mat_data)  # Float

        if self.config.is_cuda:
            self.tensor_zero = self.tensor_zero.cuda()
            self.r_head = self.r_head.cuda()
            self.r_tail = self.r_tail.cuda()
            self.r_head_sum = self.r_head_sum.cuda()
            self.r_tail_sum = self.r_tail_sum.cuda()

            self.eer_adj_index = self.eer_adj_index.cuda()
            self.eer_adj_data = self.eer_adj_data.cuda()


    # 2 relation
    def forward(self):
        # 1) model_name
        # name_embed = self.kg_name_embed
        name_embed = torch.mm(self.kg_name_embed, self.kg_name_w) + self.kg_name_b.squeeze(1)

        # 2 gat model
        r_embed_1 = self.add_r_layer(name_embed)  # (R,r_dim)
        e_embed_1 = self.add_e_att_layer(name_embed, r_embed_1)
        gat_embed_1 = name_embed + self.config.beta1 * e_embed_1
        # two layer
        r_embed_2 = self.add_r_layer(gat_embed_1)
        e_embed_2 = self.add_e_att_layer(gat_embed_1, r_embed_2)
        gat_embed = name_embed + self.config.beta1 * e_embed_2

        # GCN+highway
        if self.beg_gcn:
            gcn_embed = self.gcn_model(gat_embed)
            #e_embed_2 = ent_embed + 1.0 * e_embed_1
            return gcn_embed  # (E_new, dim)
        else:
            return gat_embed  # (E, dim)


    def add_e_att_layer(self, ent_embed, r_layer):
        e_i_layer = ent_embed[self.eer_adj_index[0, :], :]
        e_j_layer = ent_embed[self.eer_adj_index[1, :], :]
        e_ij_embed = torch.cat((e_i_layer, e_j_layer), dim=1)

        # [ei||ej]*rij        # (D,r_dim)  D = 176396
        r_qtr = r_layer[self.eer_adj_data]
        eer_embed = e_ij_embed * r_qtr  # (D,r_dim)

        # ee_atten = leakyrelu(a*eer_embed)eer_embed:
        ee_atten = torch.exp(
            -self.leakyrelu(torch.matmul(eer_embed, self.w_atten_r).squeeze()))  # (D,r_dim)*(r_dim,1) => D

        # e_rowsum => (E,E)*(E,1) = (E,1)
        dv = 'cuda' if self.config.is_cuda else 'cpu'
        e_rowsum = self.special_spmm(self.eer_adj_index, ee_atten, torch.Size([self.kg_E, self.kg_E]),
                                     torch.ones(size=(self.kg_E, 1), device=dv))  # (E,E) (E,1) => (E,1)
        e_rowsum = torch.where(e_rowsum == 0, self.tensor_zero, 1. / e_rowsum)

        # e_out: attention*h = ee_atten * e_embed => (E,E)*(E,dim) = (E,dim)
        e_out = self.special_spmm(self.eer_adj_index, ee_atten, torch.Size([self.kg_E, self.kg_E]),
                                  ent_embed)  # (E,dim)
        e_out = e_out * e_rowsum  # (E,dim)

        return self.relu(e_out)  # (E,dim)

    # add relation layer
    def add_r_layer(self, e_inlayer):
        L_e_inlayer = torch.mm(e_inlayer, self.w_R_Left)
        L_r_embed = torch.matmul(self.r_head, L_e_inlayer)  # (R,E)*(E,d) => (R,d)
        L_r_embed = L_r_embed * self.r_head_sum  # / r_head_sum => Multiply by the reciprocal instead

        R_e_inlayer = torch.mm(e_inlayer, self.w_R_Right)
        R_r_embed = torch.matmul(self.r_tail, R_e_inlayer)  # (R,E)*(E,d) => (R,d)
        R_r_embed = R_r_embed * self.r_tail_sum  # / r_tail_sum => Multiply by the reciprocal instead

        r_embed = torch.cat([L_r_embed, R_r_embed], dim=-1)  # (r,600)
        return self.relu(r_embed)  # shape=# (R,2*d)


    ## Loss function, negative sampling ################
    def get_loss(self, kg_embed, isTrain=True):
        # negative sampling
        if isTrain:
            tt_neg = self.train_neg_pairs  # (pe1, pe2, neg_indexs1[i], neg_indexs2[i])
        else:
            tt_neg = self.valid_neg_pairs

        # Positive sample
        pe1_embed = kg_embed[tt_neg[:, 0]]
        pe2_embed = kg_embed[tt_neg[:, 1]]
        A = alignment.mypair_distance_min(pe1_embed, pe2_embed, distance_type=self.config.metric)
        D = (A + self.config.gamma_rel)

        # Negative sample 1, p1 negative sample
        ne1_embed = kg_embed[tt_neg[:, 2]]
        B = alignment.mypair_distance_min(pe1_embed, ne1_embed, distance_type=self.config.metric)
        loss1 = self.relu(D - B)

        # Negative sample 2, p2 negative sample
        ne2_embed = kg_embed[tt_neg[:, 3]]
        C = alignment.mypair_distance_min(pe2_embed, ne2_embed, distance_type=self.config.metric)
        loss2 = self.relu(D - C)

        loss = torch.mean(loss1 + loss2)
        return loss


    def accuracy(self, kg_embed, type=0):
        with torch.no_grad():
            if type == 0:
                tt_links_new = np.array(self.train_links_new)
            elif type == 1:
                tt_links_new = np.array(self.valid_links_new)
            else:
                tt_links_new = np.array(self.test_links_new)

            tt_links_tensor = torch.LongTensor(tt_links_new)
            Left_vec = kg_embed[tt_links_tensor[:, 0], :]
            Right_vec = kg_embed[tt_links_tensor[:, 1], :]
            # From left
            all_hits, mr, mrr, hits1_list, noHits1_list, notin_candi = self.get_hits(Left_vec, Right_vec, tt_links_new)
            # Similarity calculation measure:
            result_str1 = "==accurate results: hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}, noHits1:{}".format(
                self.config.top_k, all_hits, mr, mrr, len(noHits1_list))

            if type == 2:
                # From left
                result_str1 = '\nFrom left ' + result_str1
                # From Right
                tt_links_new = tt_links_new[:, [1, 0]]
                all_hits2, mr2, mrr2, hits1_list2, noHits1_list2, notin_candi2 = self.get_hits(Right_vec, Left_vec, tt_links_new)
                result_str1 += "\n==From right ==accurate results: hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}, noHits1:{}".format(
                    self.config.top_k, all_hits2, mr2, mrr2, len(noHits1_list2))


        return (all_hits, mr, mrr), result_str1, hits1_list, noHits1_list

    # Accuracy, hits and other results
    def get_hits(self, Left_vec, Right_vec, tt_links_new):
        max_index = alignment.torch_sim_max_topk(Left_vec, Right_vec, top_num=self.config.top_k[-1], metric=self.config.metric)

        # left
        mr = 0
        mrr = 0
        notin_candi = 0
        tt_num = len(tt_links_new)
        all_hits = [0] * len(self.config.top_k)
        hits1_list = list()
        noHits1_list = list()
        for row_i in range(max_index.shape[0]):
            e2_ranks_index = max_index[row_i, :]
            if self.beg_gcn == False:  # Only the old id
                e1_old_gold, e2_old_gold = tt_links_new[row_i]
                e2_rank_oldids = tt_links_new[e2_ranks_index, 1].tolist()
            else:
                e1_new_gold, e2_new_gold = tt_links_new[row_i]
                e2_rank_newids = tt_links_new[e2_ranks_index, 1]
                # old ent
                e1_old_gold, e2_old_gold = self.newentid_dict[e1_new_gold], self.newentid_dict[e2_new_gold]
                e2_rank_oldids = [self.newentid_dict[new_id] for new_id in e2_rank_newids]
                e2_rank_oldids = model_util.link_inset(e2_rank_oldids)

            hits1_list.append((e1_old_gold, e2_rank_oldids[0]))
            if e2_old_gold != e2_rank_oldids[0]:
                noHits1_list.append((e1_old_gold, e2_old_gold, e2_rank_oldids[0]))
            if e2_old_gold not in e2_rank_oldids:
                notin_candi += 1
            else:
                rank_index = e2_rank_oldids.index(e2_old_gold)
                mr += (rank_index + 1)
                mrr += 1 / (rank_index + 1)
                for j in range(len(self.config.top_k)):
                    if rank_index < self.config.top_k[j]:
                        all_hits[j] += 1

        assert len(hits1_list) == tt_num
        all_hits = [round(hh / tt_num * 100, 4) for hh in all_hits]
        mr /= tt_num
        mrr /= tt_num

        return all_hits, mr, mrr, hits1_list, noHits1_list, notin_candi


    def links_old_init(self, kgs_data):
        # new neg link and links
        self.train_links_new = []
        self.test_links_new = []
        self.valid_links_new = []
        self.left_non_train, self.right_non_train = [], []

        for e1, e2 in kgs_data.train_links:
            self.train_links_new.append((e1, e2))

        for e1, e2 in kgs_data.valid_links:
            self.valid_links_new.append((e1, e2))
            self.left_non_train.append(e1)
            self.right_non_train.append(e2)

        for e1, e2 in kgs_data.test_links:
            self.test_links_new.append((e1, e2))
            self.left_non_train.append(e1)
            self.right_non_train.append(e2)


    def links_new_init(self, kgs_data, newentid_dict, ent_new_pair):
        self.newentid_dict = newentid_dict

        self.train_links_new = \
            model_util.links_init_partiton(kgs_data.train_links, ent_new_pair)
        self.valid_links_new =\
            model_util.links_init_partiton(kgs_data.valid_links, ent_new_pair)
        self.test_links_new =\
            model_util.links_init_partiton(kgs_data.test_links, ent_new_pair)

        self.left_non_train, self.right_non_train = [], []
        for e1, e2 in self.valid_links_new + self.test_links_new:
            self.left_non_train.append(e1)
            self.right_non_train.append(e2)

    # add GCN layer, only execute once
    def GCN_beg(self, ent_embed):
        self.myprint('++++++++begin GCN model+++++++++')
        self.beg_gcn = self.is_regen = True

        kgs_data = fileUtil.loadpickle(self.config.division_save + 'kgsdata.pkl')
        # Generate the new ent id set
        newentid_dict, ent_neigh_dict, ent_new_pair, ent_old2list_dict = kgs_data.pre_gen(ent_embed, self.config)
        fileUtil.savepickle(self.config.division_save + 'kgsdata2.pkl', kgs_data)

        # Reset training set
        self.links_new_init(kgs_data, newentid_dict, ent_new_pair)
        # Reset neighborhood
        self.gcn_model.data_init(newentid_dict, ent_neigh_dict)


    def GCN_beg_reset(self):
        self.myprint('++++++++begin GCN model+++++++++')
        self.beg_gcn = self.is_regen = True

        kgs_data = fileUtil.loadpickle(self.config.division_save + 'kgsdata2.pkl')
        newentid_dict, ent_neigh_dict, ent_new_pair, ent_old2list_dict = kgs_data.newentid_dict, \
                              kgs_data.ent_neigh_dict, kgs_data.ent_new_pair, kgs_data.ent_old2list_dict

        # Reset training set
        self.links_new_init(kgs_data, newentid_dict, ent_new_pair)  # , ent_old2list_dict
        # Reset neighborhood
        self.gcn_model.data_init(newentid_dict, ent_neigh_dict)


    def regen_neg(self, epochs_i, ent_embed):
        if self.is_regen or epochs_i % 20 == 0:
            self.train_neg_pairs = model_util.gen_neg(ent_embed, self.train_links_new,
                                                      self.config.metric, self.config.neg_k)
            self.valid_neg_pairs = model_util.gen_neg(ent_embed, self.valid_links_new,
                                                      self.config.metric, self.config.neg_k)
            if self.is_regen:
                self.is_regen = False


    ## Iter ####################
    def reset_IIL(self, new_ILL_list):
        self.myprint('++++++Reset IIL by Iter +++++++++')
        self.is_regen = True
        self.myprint('new_ILL_list len:' + str(len(new_ILL_list)))

        new_links = self.train_links_new + new_ILL_list
        self.train_links_new = new_links
        self.myprint('train_links_new len:' + str(len(new_links)))

        for nl in new_ILL_list:
            self.left_non_train.remove(nl[0])
            self.right_non_train.remove(nl[1])
        self.myprint('left_non_train len:' + str(len(self.left_non_train)))


    def get_newIIL(self, ent_embed, new_links=[]):

        left_tensor = torch.LongTensor(self.left_non_train)
        right_tensor = torch.LongTensor(self.right_non_train)
        if ent_embed.is_cuda:
            left_tensor = left_tensor.cuda()
            right_tensor = right_tensor.cuda()

        ent_embed_left = ent_embed[left_tensor]
        ent_embed_right = ent_embed[right_tensor]

        index_mat_left, score_mat_left = alignment.torch_sim_max_topk_s(ent_embed_left, ent_embed_right, top_num=1, metric='cosine')
        index_mat_left, score_mat_left = index_mat_left.squeeze(-1), score_mat_left.squeeze(-1)

        index_mat_right, score_mat_right = alignment.torch_sim_max_topk_s(ent_embed_right, ent_embed_left, top_num=1, metric='cosine')
        index_mat_right, score_mat_right = index_mat_right.squeeze(-1), score_mat_right.squeeze(-1)

        ILL_list = []
        for i, p in enumerate(index_mat_left):
            # Two-way alignment
            if score_mat_left[i]<0.95 or score_mat_right[p]<0.95:
                continue
            elif index_mat_right[p] == i:
                ee_pair = (self.left_non_train[i], self.right_non_train[p])
                if new_links == [] or (ee_pair in new_links): # Intersection
                    ILL_list.append(ee_pair)

        if len(ILL_list) > 0:
            noinILL = len(set(ILL_list) - set(self.valid_links_new + self.test_links_new))
            inILL = len(ILL_list) - noinILL
            self.myprint("==True ILL({})/ all IIL({}):{:.4f}%, == non_train:{}. ".format(inILL, len(ILL_list),
                                          inILL / len(ILL_list) * 100, len(self.left_non_train)))
        else:
            self.myprint("==no IIL")
        return ILL_list
