import os
import sys
import time
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from align import pre_loadKGs
from align_noname.model_union_noName import align_union_noName
from autil import fileUtil

class align_union_set100K():
    def __init__(self, config):
        super(align_union_set100K, self).__init__()
        self.config = config
        self.myprint = config.myprint
        self.best_mode_pkl_title = config.output + config.time_str
        # 加载基础数据 Load KGs data
        kgsdata_file = config.division_save + 'kgsdata.pkl'
        if os.path.exists(kgsdata_file):
            kgs_data = fileUtil.loadpickle(kgsdata_file)
        else:
            kgs_data = pre_loadKGs.load_KGs_data(config)
            fileUtil.savepickle(kgsdata_file, kgs_data)
        self.myprint('====Finish Load kgs_data====')

        # 定义KGs 模型#######################
        self.kg_Model = align_union_noName(kgs_data, config)
        if config.is_cuda:
            self.kg_Model = self.kg_Model.cuda()

        ## Model and optimizer ######################
        if config.optim_type == 'Adagrad':
            self.optimizer = optim.Adam(self.kg_Model.model_params, lr=config.learning_rate,
                            weight_decay=config.weight_decay)  # 权重衰减（参数L2损失）weight_decay =5e-4
        else:
            self.optimizer = optim.SGD(self.kg_Model.model_params, lr=config.learning_rate,
                            weight_decay=config.weight_decay)
        self.myprint('model中所有参数名:' + str(len(self.kg_Model.state_dict()) + 1))
        for i in self.kg_Model.state_dict():  # 打印model中所有参数名。
            self.myprint(i)

        ##迭代学习
        self.isGCN = config.isGCN
        if config.isGCN:  # 有迭代的
            self.myprint("叠加GCN学习==")
        else:
            self.myprint("无GCN学习==")
        ##迭代学习
        if config.isIter:  # 有迭代的
            self.myprint("有迭代学习==")
            self.ILL_temp = []# 临时集和固定集
            self.iter_forever_count = 0 # 5*4=20

        ## noname!
        self.myprint("无name embed ==")
        self.beg_test_hits1 = 55.0 # 100K

    ## model train
    def model_train(self, beg_epochs=0):
        t_begin = time.time()
        # 【TensorboardX】
        #self.board_writer = SummaryWriter(log_dir=self.best_mode_pkl_title + '-E/', comment='HET_align')
        self.myprint("model training start==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))

        self.bad_counter, self.best_hits1, self.best_epochs = 0, 0, -1  # best
        beg_bad = 0
        for epochs_i in range(beg_epochs, self.config.train_epochs):  # epochs=1001
            ## 训练
            kg_ent_emb = self.runTrain(epochs_i)

            ## 验证
            if epochs_i >= self.config.start_valid and epochs_i % self.config.eval_freq == 0:
                break_re, test_hits1 = self.runTest(epochs_i)
                if self.config.early_stop and break_re:
                    break

                # 叠加GCN，在准确率达到一定程度后
                if self.isGCN and self.kg_Model.beg_gcn == False:
                    if test_hits1 > self.beg_test_hits1:
                        self.kg_Model.GCN_beg(kg_ent_emb)  # 开始gcn操作！！
                    else:
                        beg_bad += 1

                    if beg_bad > 20: # 最坏打算
                        print('===bad GCN_beg')
                        self.kg_Model.GCN_beg(kg_ent_emb)  # 开始gcn操作！！


        # 输出相关数据
        self.save_model(epochs_i, 'last')  # 保存 last_epochs
        self.myprint("Optimization Finished!")
        self.myprint('Best epoch-{:04d}:'.format(self.best_epochs))
        self.myprint('Last epoch-{:04d}:'.format(epochs_i))
        self.myprint("Total time elapsed: {:.4f}s".format(time.time() - t_begin))

        return self.best_epochs, epochs_i

    ## 迭代 ################
    def Iter_reset(self, epochs_i, ent_embed):
        with torch.no_grad():
            if epochs_i % 10 == 0:  # 迭代iter_beg开始，每k=5 轮，更新一次，每10*5更新固定对齐集
                # 临时集
                if len(self.ILL_temp) == 0:
                    self.iter_forever_count = 0
                self.ILL_temp = self.kg_Model.get_newIIL(ent_embed, self.ILL_temp) # 交集，持续在临时集
                self.iter_forever_count += 1
                if self.iter_forever_count == 5 and len(self.ILL_temp) > 0:
                    self.kg_Model.reset_IIL(self.ILL_temp)  # 重置  训练集
                    self.ILL_temp = []

    ## 运行每轮训练
    def runTrain(self, epochs_i):
        t_epoch = time.time()
        # 训练 Forward pass
        self.kg_Model.train()
        self.optimizer.zero_grad()  # 梯度清零

        # Model trainning
        kg_ent_emb = self.kg_Model()
        ## 迭代
        if self.config.isIter and self.kg_Model.beg_gcn:  #
            self.Iter_reset(epochs_i, kg_ent_emb)  #
        ## 重新负采样
        self.kg_Model.regen_neg_noName(epochs_i, kg_ent_emb) ## regen_neg, regen_neg_noName!

        # loss
        train_loss = self.kg_Model.get_loss(kg_ent_emb, isTrain=True)
        loss_float = train_loss.data.item()

        # Backward and optimize
        train_loss.backward()  # retain_graph=True 多个loss的自定义loss
        self.optimizer.step()  # 更新权重参数

        # 计算准确率
        result_str1 = ''
        if epochs_i % 5 == 0:
            hits_mr_mrr, result_str1, _, _ = self.kg_Model.accuracy(kg_ent_emb, type=0)

            #self.board_writer.add_scalar('Train_hits1', hits_mr_mrr[0][0], epochs_i)
            # # 叠加GCN，在准确率达到一定程度后
            # if self.isGCN and self.kg_Model.beg_gcn == False and hits_mr_mrr[0][0] > self.beg_GCN_thres:
            #     self.kg_Model.GCN_beg(kg_ent_emb)  # 开始gcn操作！！

        # 输出结果
        # [TensorboardX]
        #self.board_writer.add_scalar('Train_loss', loss_float, epochs_i)

        self.myprint('Epoch-{:04d}: Train_loss-{:.8f}, cost time-{:.4f}s'.format(
            epochs_i, loss_float, time.time() - t_epoch))
        if result_str1 != '':
            self.myprint('==Train' + result_str1)  # 准确率等
        return kg_ent_emb


    ## 运行测试
    def runTest(self, epochs_i=0, isSave=False):
        rebreak = False
        with torch.no_grad():
            # Forward pass
            self.kg_Model.eval()

            # 获得loss，最终嵌入
            kg_ent_emb = self.kg_Model()
            hits_mr_mrr, result_str1, hits1_list, noHits1_list = self.kg_Model.accuracy(kg_ent_emb, type=2)
            self.myprint('==Test' + result_str1)  # 准确率等
            if hits_mr_mrr[0][0] >= self.best_hits1:  # 验证集的准确率，连续没有提高
                self.best_hits1 = hits_mr_mrr[0][0]  # 损失效果比之前的好，hits1 ？
                self.best_epochs = epochs_i
                self.bad_counter = 0
                self.myprint('Epoch-{:04d}, better result, best_hits1:{:.4f}..'.format(epochs_i, self.best_hits1))
                self.save_model(epochs_i, 'best')  # 保存最好的模型
            else:
                self.bad_counter += 1
                self.myprint('==bad_counter++:' + str(self.bad_counter))
                # bad model, stop train
                if self.bad_counter == self.config.patience:  # patience=20
                    self.myprint('==bad_counter, stop training.')
                    rebreak = True

            # 保存到文档
            if isSave:
                model_result_file = '{}_Result.txt'.format(self.best_mode_pkl_title)
                with open(model_result_file, "a") as ff:
                    ff.write('Test' + result_str1 + '\n')

                # 输出测试集结果：
                fileUtil.save_list2txt('{}_test_hits1_list.txt'.format(self.best_mode_pkl_title), hits1_list)
                fileUtil.save_list2txt('{}_test_noHits1_list.txt'.format(self.best_mode_pkl_title), noHits1_list)

                # 输出验证集结果：
                hits_mr_mrr, result_str1, hits1_list, noHits1_list = self.kg_Model.accuracy(kg_ent_emb, type=1)
                fileUtil.save_list2txt('{}_valid_hits1_list.txt'.format(self.best_mode_pkl_title), hits1_list)
                fileUtil.save_list2txt('{}_valid_noHits1_list.txt'.format(self.best_mode_pkl_title), noHits1_list)

                # 保存嵌入结果！
                # savefile = '{}kg_ent_emb.emb'.format(self.best_mode_pkl_title)
                # print('savefile:', savefile)
                # fileUtil.savepickle(savefile, kg_ent_emb)

        return rebreak, hits_mr_mrr[0][0]


    ## 模型保存，重新测试，输出结果 #############
    def reRunTest(self, better_epochs_i, epochs_name, isSave=False):
        ''' run best model '''
        model_savefile = '{}-epochs-{}-{}.pkl'.format(self.best_mode_pkl_title, better_epochs_i, epochs_name)
        self.myprint('\nLoading {} - {}th epoch'.format(epochs_name, better_epochs_i))
        self.runTestFile(model_savefile, isSave=isSave, is_cuda=self.config.is_cuda)


    def runTestFile(self, model_savefile, isSave=False, is_cuda=False):
        ''' restart run best model '''
        self.myprint('Loading file: ' + model_savefile)
        # 加载模型
        if is_cuda:
            checkpoint = torch.load(model_savefile)
        else:
            checkpoint = torch.load(model_savefile, map_location='cpu')  # GPU->CPU

        self.kg_Model.load_state_dict(checkpoint['kg_layer'])
        # 运行测试
        self.runTest(isSave=isSave)


    def save_model(self, better_epochs_i, epochs_name):  # best-epochs
        # save model to file ：文件名开头，参数列表
        model_savefile = '{}-epochs-{}-{}.pkl'.format(self.best_mode_pkl_title, better_epochs_i, epochs_name)
        # 分别保存模型
        model_state = dict()
        model_state['kg_layer'] = self.kg_Model.state_dict()
        torch.save(model_state, model_savefile)


    def reRunTrain(self, model_savefile, beg_epochs, is_cuda=False, beg_gcn=True):
        ''' restart run train model '''
        self.myprint('Loading file: ' + model_savefile)
        # 加载模型
        if is_cuda:
            checkpoint = torch.load(model_savefile)
        else:
            checkpoint = torch.load(model_savefile, map_location='cpu')  # GPU->CPU

        self.kg_Model.load_state_dict(checkpoint['kg_layer'])
        # 运行测试
        if beg_gcn:
            self.isGCN = True
            self.kg_Model.beg_gcn = True
            self.kg_Model.GCN_beg_reset()  # 开始gcn操作！！
        best_epochs, last_epochs = self.model_train(beg_epochs=beg_epochs)
        return best_epochs, last_epochs

    ## 学习率衰减
    def adjust_learning_rate_(self, epochs):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs
        如何每30个epoch按10%的速率衰减  """
        milestones = [0, 100, 300, 500, 800, 1000, 1200, 1500]
        if epochs in milestones:
            index = milestones.index(epochs)
            lr = self.config.learning_rate * (0.4 ** index)  # 0.4^5 *0.005=0.00005
            self.myprint('==lr_{}:{}'.format(index, lr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        # 学习率衰减
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
        #                milestones=[100, 300, 500, 800, 1000, 1200, 1500], gamma=0.2)
        # self.scheduler.step()

