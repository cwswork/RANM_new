import os
import time
import torch
from align_100K.model_set_100K import align_union_set_100K
from autil.configUtil import configClass


def run(datasets, division, model_type='addIter', is_cuda=True):
    print('\n\n++++++++++++++++++++++++++++++++++++')

    myconfig = configClass('../args_15K.json', datasets=datasets, division=division)
    # myconfig.seed = 100
    myconfig.is_cuda = is_cuda and torch.cuda.is_available()  # cuda是否可用
    myconfig.batch_size = 1024 * 20
    ###### 模型选择
    #model_type = 'addIter'  # noGCN, addGCN, addIter
    myconfig.output += 'neigh(100K)_' + model_type + '/' + myconfig.time_str + '/'
    if model_type == 'noGCN':
        myconfig.isGCN = False  # 是否加GCN, 默认True
    if model_type == 'addIter':
        myconfig.isIter = True  # 是否迭代，默认False
        myconfig.patience_minloss = 40

    myconfig.gamma_rel = 10.0
    myconfig.learning_rate = 0.001
    myconfig.eval_freq = 10
    myconfig.start_valid = 50
    myconfig.neg_k = 25

    #########  定义模型
    myconfig.set_myprint(os.path.realpath(__file__))  # 初始化, 打印和日志记录
    mymodel = align_union_set_100K(myconfig)
    # 模型训练
    myconfig.myprint("\n===train align_model")
    best_epochs, last_epochs = mymodel.model_train()
    # 最后模型测试
    mymodel.myprint("===Run test on last_epochs: ")
    mymodel.runTest(last_epochs, isSave=True)  # 运行最后的模型
    # 最好模型测试
    mymodel.myprint("===Run test on best_epochs: ")
    mymodel.reRunTest(best_epochs, 'best', isSave=True)  # 运行测试集 Testing
    mymodel.myprint("end==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    print('---------------------------------------------')

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    is_cuda = False
    # DWY100K/dbp_wd  DWY100K/dbp_yg
    # dataset = 'DWY100K/dbp_wd'
    # # noGCN, addGCN, addIter
    # print("\n####begin, " + dataset + ", addIter==noGCN" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    # run(datasets=dataset + '/', division='1/', model_type='noGCN')
    #
    # print("\n####begin, " + dataset + ", addIter==addGCN" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    # run(datasets=dataset + '/', division='1/', model_type='addGCN')
    #
    # print("\n####begin, " + dataset + ", addIter==addIter" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    # run(datasets=dataset + '/', division='1/', model_type='addIter')

    dataset = 'DWY100K/dbp_wd'
    #print("\n####begin, " + dataset + ", addIter==noGCN" + time.strftime('%Y.%m.%d %H:%M:%S',                                                                 time.localtime(time.time())))
    #run(datasets=dataset + '/', division='1/', model_type='noGCN', is_cuda=is_cuda)

    print("\n####begin, " + dataset + ", addIter==addGCN" + time.strftime('%Y.%m.%d %H:%M:%S',                                                                  time.localtime(time.time())))
    run(datasets=dataset + '/', division='1/', model_type='addGCN', is_cuda=is_cuda)

    print("\n####begin, " + dataset + ", addIter==addIter" + time.strftime('%Y.%m.%d %H:%M:%S',                                                          time.localtime(time.time())))
    run(datasets=dataset + '/', division='1/', model_type='addIter', is_cuda=is_cuda)


