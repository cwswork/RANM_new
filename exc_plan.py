import os
import time
import torch
from align.model_set import align_union_set
from autil.configUtil import configClass


def run(datasets, division, model_type='addIter', is_cuda=True):
    print('\n\n++++++++++++++++++++++++++++++++++++')

    # args
    myconfig = configClass('args_15K.json', datasets=datasets, division=division)
    # GPU
    myconfig.is_cuda = is_cuda and torch.cuda.is_available()  # cuda
    myconfig.batch_size = 1024 * 20
    ###### Model Choice
    myconfig.output += 'neigh(07.27)_' + model_type + '/' + myconfig.time_str + '/'
    if model_type == 'noGCN':
        myconfig.isGCN = False  # add GCN, Defalut True
    if model_type == 'addIter':
        myconfig.isIter = True  # add Iter，Defalut False

    # myconfig.patience_minloss = 40
    # myconfig.patience = 40
    # myconfig.gamma_rel = 10.0  # 5, 10, 15
    # myconfig.learning_rate = 0.001
    #########   Model Definition
    myconfig.set_myprint(os.path.realpath(__file__)) # print
    mymodel = align_union_set(myconfig)
    # Model Training
    myconfig.myprint("\n===train align_model")
    best_epochs, last_epochs = mymodel.model_train()
    # Model Test
    mymodel.myprint("===Run test on last_epochs: ")
    mymodel.runTest(isSave=True)
    # Best Model Test
    mymodel.myprint("===Run test on best_epochs: ")
    mymodel.reRunTest(best_epochs, 'best', isSave=True)
    mymodel.myprint("end==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    print('---------------------------------------------')

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    is_cuda = False
    dataset = 'EN_DE_15K_V1' # EN_DE_15K_V1, EN_FR_15K_V1, fr_en(dbp15) ja_en(dbp15), zh_en(dbp15) 、D_Y_100K_V1
    # noGCN, addGCN, addIter
    # print("\n####begin, " + dataset + ", addIter==noGCN" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    # run(datasets=dataset + '/', division='1/', model_type='noGCN')
    # run(datasets=dataset + '/', division='2/', model_type='noGCN')
    # run(datasets=dataset + '/', division='3/', model_type='noGCN')
    # run(datasets=dataset + '/', division='4/', model_type='noGCN')
    # run(datasets=dataset + '/', division='5/', model_type='noGCN')
    # #
    # print("\n####begin, " + dataset + ", addIter==addGCN" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    # run(datasets=dataset + '/', division='1/', model_type='addGCN')
    # run(datasets=dataset + '/', division='2/', model_type='addGCN')
    # run(datasets=dataset + '/', division='3/', model_type='addGCN')
    # run(datasets=dataset + '/', division='4/', model_type='addGCN')
    # run(datasets=dataset + '/', division='5/', model_type='addGCN')

    print("\n####begin, " + dataset + ", addIter==addIter" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    run(datasets=dataset + '/', division='1/', model_type='addIter')
    run(datasets=dataset + '/', division='2/', model_type='addIter')
    run(datasets=dataset + '/', division='3/', model_type='addIter')
    run(datasets=dataset + '/', division='4/', model_type='addIter')
    run(datasets=dataset + '/', division='5/', model_type='addIter')

