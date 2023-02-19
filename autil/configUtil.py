from __future__ import division
from __future__ import print_function
import os
import random
import re
import time

import json
import numpy as np
import torch

class Myprint:
    def __init__(self, filePath, filename):
        if not os.path.exists(filePath):
            print('output not exists' + filePath)
            os.makedirs(filePath)

        self.outfile = filePath + filename

    def print(self, print_str):
        print(print_str)
        '''Save log file'''
        with open(self.outfile, 'a', encoding='utf-8') as fw:
            fw.write('{}\n'.format(print_str))


class configClass():
    def __init__(self, args_file, datasets, division):
        args = Load_args(args_file)
        self.datasetPath = args.datasetPath + datasets+ 'pre4/'
        #self.division = division
        self.dataset_division = '721_5fold/' + division
        self.output = args.output + datasets + division
        # pre

        # embed file
        self.division_save = self.datasetPath + division  #
        if not os.path.exists(self.division_save):
            os.makedirs(self.division_save)
        #
        if not os.path.exists(self.division_save + 'temp/'):
            os.makedirs(self.division_save + 'temp/')

        self.seed = args.seed
        self.time_str = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time()))
        #
        self.attn_heads = 1
        self.rel_dim = args.rel_dim

        #
        self.optim_type = args.optim_type
        self.patience = args.patience
        self.patience_minloss = args.patience_minloss
        self.metric = args.metric
        self.train_epochs = args.train_epochs
        self.is_cuda = True
        self.isGCN = True  #
        self.isIter = False  #
        self.Name_embed = True
        #
        self.early_stop = args.early_stop
        self.start_valid = args.start_valid  #
        self.eval_freq = args.eval_freq   #
        self.eval_save_freq = args.eval_save_freq  #  20

        #
        self.top_k = args.top_k
        self.neg_k = args.neg_k

        #
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.LeakyReLU_alpha = args.LeakyReLU_alpha
        self.dropout = args.dropout
        self.gamma_rel = args.gamma_rel
        self.beta1 = args.beta1

    #
    def get_param(self):
        self.model_param = 'epochs_' + str(self.train_epochs) + \
             '-negk_' + str(self.neg_k) + \
             '-dis_' + str(self.metric) + \
             '-lr_' + str(self.learning_rate) + \
             '-reldim_' + str(self.rel_dim) + \
             '-be1_' + str(self.beta1) + \
           '-attn_' + str(self.attn_heads) + \
           '-drop_' + str(self.dropout)

        self.model_param += '-garel_' + str(self.gamma_rel)

        #
        self.model_param += '-isIter_' + str(self.isIter)
        #
        self.model_param += '-isGCN_' + str(self.isGCN)

        return self.model_param

    def set_myprint(self, runfile, issave=True):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if issave:
            print_Class = Myprint(self.output, 'train_log' + self.time_str + '.txt')
            if not os.path.exists(self.output):
                print('output not exists' + self.output)
                os.makedirs(self.output)
            self.myprint = print_Class.print
        else:
            self.myprint = print

        self.myprint("start==" + self.time_str + ": " + runfile)
        self.myprint('output Path:' + self.output)
        self.myprint('cuda.is_available:' + str(self.is_cuda))
        self.myprint('model arguments:' + self.get_param())


def Load_args(file_path):
    '''  args/** .json '''
    args_dict = loadmyJson(file_path)  #
    # print("load arguments:", args_dict)
    args = ARGs(args_dict)
    return args


class ARGs:
    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)


################################################
# Load JSON File
def loadmyJson(JsonPath):
    try:
        srcJson = open(JsonPath, 'r', encoding= 'utf-8')
    except:
        print('cannot open ' + JsonPath)
        quit()

    dstJsonStr = ''
    for line in srcJson.readlines():
        if not re.match(r'\s*//', line) and not re.match(r'\s*\n', line):
            dstJsonStr += cleanNote(line)

    # print dstJsonStr
    dstJson = {}
    try:
        dstJson = json.loads(dstJsonStr)
    except:
        print(JsonPath + ' is not a valid json file')

    return dstJson

def cleanNote(line_str):
    qtCnt = cmtPos = 0

    rearLine = line_str
    while rearLine.find('//') >= 0: # find “//”
        slashPos = rearLine.find('//')
        cmtPos += slashPos
        headLine = rearLine[:slashPos]
        while headLine.find('"') >= 0:
            qtPos = headLine.find('"')
            if not isEscapeOpr(headLine[:qtPos]):
                qtCnt += 1
            headLine = headLine[qtPos+1:]
        if qtCnt % 2 == 0:
            return line_str[:cmtPos]
        rearLine = rearLine[slashPos+2:]
        cmtPos += 2

    return line_str


#
def isEscapeOpr(instr):
    if len(instr) <= 0:
        return False
    cnt = 0
    while instr[-1] == '\\':
        cnt += 1
        instr = instr[:-1]
    if cnt % 2 == 1:
        return True
    else:
        return False
