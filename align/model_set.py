import os
import sys
import time
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from align import pre_loadKGs
from align.model_union import align_union
from autil import fileUtil

class align_union_set():
    def __init__(self, config):
        super(align_union_set, self).__init__()
        self.config = config
        self.myprint = config.myprint
        self.best_mode_pkl_title = config.output + config.time_str
        # Load KGs data
        kgsdata_file = config.division_save + 'kgsdata.pkl'
        if os.path.exists(kgsdata_file):
            kgs_data = fileUtil.loadpickle(kgsdata_file)
        else:
            kgs_data = pre_loadKGs.load_KGs_data(config)
            fileUtil.savepickle(kgsdata_file, kgs_data)
        self.myprint('====Finish Load kgs_data====')

        #  Model Definition#######################
        self.kg_Model = align_union(kgs_data, config)
        if config.is_cuda:
            self.kg_Model = self.kg_Model.cuda()

        ## Model and optimizer ######################
        if config.optim_type == 'Adagrad':
            self.optimizer = optim.Adam(self.kg_Model.model_params, lr=config.learning_rate,
                            weight_decay=config.weight_decay)  # weight_decay =5e-4
        else:
            self.optimizer = optim.SGD(self.kg_Model.model_params, lr=config.learning_rate,
                            weight_decay=config.weight_decay)
        self.myprint('All parameter names in the model:' + str(len(self.kg_Model.state_dict()) + 1))
        for i in self.kg_Model.state_dict():
            self.myprint(i)

        self.isGCN = config.isGCN
        if config.isGCN:
            self.myprint("add GCN==")
        else:
            self.myprint("no GCN==")
        if config.isIter:  # Iter
            self.myprint("Iter==")
            self.ILL_temp = []
            self.iter_forever_count = 0
        else:
            self.myprint("no Iter==")

    ## model train
    def model_train(self, beg_epochs=0):
        t_begin = time.time()
        # TensorboardX
        self.board_writer = SummaryWriter(log_dir=self.best_mode_pkl_title + '-E/', comment='HET_align')
        self.myprint("model training start==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))

        self.bad_counter, self.min_validloss_counter = 0, 0
        self.best_hits1, self.best_epochs = 0, -1  # best
        self.min_valid_loss = sys.maxsize
        for epochs_i in range(beg_epochs, self.config.train_epochs):
            self.runTrain(epochs_i)

            ## valid
            if epochs_i >= self.config.start_valid and epochs_i % self.config.eval_freq == 0:
                break_re = self.runValid(epochs_i)
                if self.config.early_stop and break_re:
                    break

        # Output related data
        self.save_model(epochs_i, 'last')  #  last_epochs
        self.myprint("Optimization Finished!")
        self.myprint('Best epoch-{:04d}:'.format(self.best_epochs))
        self.myprint('Last epoch-{:04d}:'.format(epochs_i))
        self.myprint("Total time elapsed: {:.4f}s".format(time.time() - t_begin))

        return self.best_epochs, epochs_i

    def Iter_reset(self, epochs_i, ent_embed):
        with torch.no_grad():
            if epochs_i % 10 == 0:
                # Temporary set
                if len(self.ILL_temp) == 0:
                    self.iter_forever_count = 0
                self.ILL_temp = self.kg_Model.get_newIIL(ent_embed, self.ILL_temp) # Intersection, continuous in the temporary set
                self.iter_forever_count += 1
                if self.iter_forever_count == 5 and len(self.ILL_temp) > 0:
                    self.kg_Model.reset_IIL(self.ILL_temp)  # Reset training set
                    self.ILL_temp = []

    ## Run each epochs of training
    def runTrain(self, epochs_i):
        t_epoch = time.time()
        # Forward pass
        self.kg_Model.train()
        self.optimizer.zero_grad()  # Gradient clear

        # Model trainning
        kg_ent_emb = self.kg_Model()
        ## Iter
        if self.config.isIter and self.kg_Model.beg_gcn:  #
            self.Iter_reset(epochs_i, kg_ent_emb)  #
        ## Re-negative sampling
        self.kg_Model.regen_neg(epochs_i, kg_ent_emb)

        # loss
        train_loss = self.kg_Model.get_loss(kg_ent_emb, isTrain=True)
        loss_float = train_loss.data.item()

        # Backward and optimize
        train_loss.backward()
        self.optimizer.step()

        # Calculation accuracy
        result_str1 = ''
        if epochs_i % 5 == 0:
            hits_mr_mrr, result_str1, _, _ = self.kg_Model.accuracy(kg_ent_emb, type=0)
            # [TensorboardX]
            self.board_writer.add_scalar('Train_hits1', hits_mr_mrr[0][0], epochs_i)

            # Add GCN layer, after the accuracy rate reaches a certain level
            if self.isGCN and self.kg_Model.beg_gcn == False and hits_mr_mrr[0][0] > 95.0:
                self.kg_Model.GCN_beg(kg_ent_emb)

        # [TensorboardX]
        self.board_writer.add_scalar('Train_loss', loss_float, epochs_i)
        self.myprint('Epoch-{:04d}: Train_loss-{:.8f}, cost time-{:.4f}s'.format(
            epochs_i, loss_float, time.time() - t_epoch))
        if result_str1 != '':
            self.myprint('==Train' + result_str1)


    ## Run every epochs of verification
    def runValid(self, epochs_i):
        t_epoch = time.time()

        with torch.no_grad():
            # Forward pass
            self.kg_Model.eval()
            # Model trainning
            kg_ent_emb = self.kg_Model()
            # loss
            valid_loss = self.kg_Model.get_loss(kg_ent_emb, isTrain=False)
            loss_float = valid_loss.data.item()
            hits_mr_mrr, result_str1, _, _ = self.kg_Model.accuracy(kg_ent_emb, type=1)

            self.myprint('Epoch-{:04d}: Valid_loss-{:.8f}, cost time-{:.4f}s'.format(
                epochs_i, loss_float, time.time() - t_epoch))
            self.myprint('==Valid' + result_str1)

            # [TensorboardX]
            self.board_writer.add_scalar('Valid_loss', loss_float, epochs_i)
            self.board_writer.add_scalar('Valid_hits1', hits_mr_mrr[0][0], epochs_i)

            # ********************no early stop********************************************
            # save best model in valid
            if hits_mr_mrr[0][0] >= self.best_hits1:
                self.best_hits1 = hits_mr_mrr[0][0]
                self.best_epochs = epochs_i
                self.bad_counter = 0
                self.myprint('Epoch-{:04d}, better result, best_hits1:{:.4f}..'.format(epochs_i, self.best_hits1))
                self.save_model(epochs_i, 'best')
            else:
                # no best, but save model every 10 epochs
                if epochs_i % self.config.eval_save_freq == 0:
                    self.save_model(epochs_i, 'eval')

                self.bad_counter += 1
                self.myprint('==bad_counter++:' + str(self.bad_counter))
                # bad model, stop train
                if self.bad_counter == self.config.patience:
                    self.myprint('==bad_counter, stop training.')
                    return True

            # Verification set loss continuous decline also stop training!
            if loss_float <= self.min_valid_loss:
                self.min_valid_loss = loss_float
                self.min_validloss_counter = 0
                self.myprint('Epoch-{:04d}, min_valid_loss:{:.8f}..'.format(epochs_i, self.min_valid_loss))
            else:
                self.min_validloss_counter += 1
                self.myprint('==min_validloss_counter++:{}'.format(self.min_validloss_counter))
                if self.min_validloss_counter == self.config.patience_minloss:
                    self.myprint('==bad min_valid_loss, stop training.')
                    return True

            return False

    ## Run test
    def runTest(self, isSave=False):
        with torch.no_grad():
            # Forward pass
            self.kg_Model.eval()

            kg_ent_emb = self.kg_Model()
            hits_mr_mrr, result_str1, hits1_list, noHits1_list = self.kg_Model.accuracy(kg_ent_emb, type=2)
            self.myprint('==Test' + result_str1)
            # Save to document
            if isSave:
                model_result_file = '{}_Result.txt'.format(self.best_mode_pkl_title)
                with open(model_result_file, "a") as ff:
                    ff.write('Test' + result_str1 + '\n')

    def reRunTest(self, better_epochs_i, epochs_name, isSave=False):
        ''' Save the model, retest, and output the result '''
        model_savefile = '{}-epochs-{}-{}.pkl'.format(self.best_mode_pkl_title, better_epochs_i, epochs_name)
        self.myprint('\nLoading {} - {}th epoch'.format(epochs_name, better_epochs_i))
        self.runTestFile(model_savefile, isSave=isSave, is_cuda=self.config.is_cuda)


    def runTestFile(self, model_savefile, isSave=False, is_cuda=False):
        ''' restart run best model '''
        self.myprint('Loading file: ' + model_savefile)
        if is_cuda:
            checkpoint = torch.load(model_savefile)
        else:
            checkpoint = torch.load(model_savefile, map_location='cpu')  # GPU->CPU

        self.kg_Model.load_state_dict(checkpoint['kg_layer'])
        self.runTest(isSave=isSave)


    def save_model(self, better_epochs_i, epochs_name):  # best-epochs
        # save model to file
        model_savefile = '{}-epochs-{}-{}.pkl'.format(self.best_mode_pkl_title, better_epochs_i, epochs_name)
        model_state = dict()
        model_state['kg_layer'] = self.kg_Model.state_dict()
        torch.save(model_state, model_savefile)


    def reRunTrain(self, model_savefile, beg_epochs, is_cuda=False, beg_gcn=True):
        ''' restart run train model '''
        self.myprint('Loading file: ' + model_savefile)
        if is_cuda:
            checkpoint = torch.load(model_savefile)
        else:
            checkpoint = torch.load(model_savefile, map_location='cpu')  # GPU->CPU

        self.kg_Model.load_state_dict(checkpoint['kg_layer'])
        if beg_gcn:
            self.isGCN = True
            self.kg_Model.beg_gcn = True
            self.kg_Model.GCN_beg_reset()
        best_epochs, last_epochs = self.model_train(beg_epochs=beg_epochs)
        return best_epochs, last_epochs
