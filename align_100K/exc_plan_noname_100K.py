import os
import time
import torch

from align_100K.model_set_noName_100K import align_union_set100K
from autil.configUtil import configClass


def runNoName(datasets, division, model_type='addIter', is_cuda=True, arg=None):
    print('\n++++++++++++, arg:', arg)

    # 创建参数配置对象 args
    myconfig = configClass('../args_15K.json', datasets=datasets, division=division)
    # myconfig.seed = 100
    # 运行设置
    myconfig.is_cuda = is_cuda and torch.cuda.is_available()  # cuda是否可用
    myconfig.batch_size = 1024 * 15
    ###### 模型选择
    #model_type = 'addIter'  # noGCN, addGCN, addIter
    myconfig.output += 'neigh(10.28)_' + model_type + '/' + myconfig.time_str + '/'
    if model_type == 'noGCN':
        myconfig.isGCN = False  # 是否加GCN, 默认True
    if model_type == 'addIter':
        myconfig.isIter = True  # 是否迭代，默认False
        myconfig.patience_minloss = 40
    myconfig.patience = 20

    # runNoName
    myconfig.start_valid = 10
    myconfig.neg_k = 125
    # myconfig.eval_freq = 20
    myconfig.attn_heads = arg[0] #2
    myconfig.rel_dim = arg[1] #100
    myconfig.dropout = arg[2] #0.3  # 丢弃的概率
    myconfig.gamma_rel = arg[3] #10.0  # 5, 10, 15
    myconfig.learning_rate = 0.0005
    #myconfig.metric = 'L1' # L1

    print()
    #########  定义模型
    myconfig.set_myprint(os.path.realpath(__file__))  # 初始化, 打印和日志记录
    mymodel = align_union_set100K(myconfig)
    # 模型训练
    myconfig.myprint("\n===train align_model")
    best_epochs, last_epochs = mymodel.model_train()
    # 最后模型测试
    mymodel.myprint("===Run test on last_epochs: ")
    mymodel.runTest(isSave=True)  # 运行最后的模型
    # 最好模型测试
    mymodel.myprint("===Run test on best_epochs: ")
    mymodel.reRunTest(best_epochs, 'best', isSave=True)  # 运行测试集 Testing
    mymodel.myprint("end==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    print('---------------------------------------------')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    is_cuda = False
    dataset = 'DWY100K/dbp_wd' # EN_DE_15K_V1, EN_FR_15K_V1, fr_en(dbp15) ja_en(dbp15), zh_en(dbp15)
    # DWY100K/dbp_wd  DWY100K/dbp_yg
    # noGCN, addGCN, addIter

    myarg = [2, 100, 0.1, 10]
    print("\n####begin, " + dataset + ", model_type==addGCN" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    runNoName(datasets=dataset + '/', division='1/', model_type='addGCN', is_cuda=is_cuda, arg=myarg)
    print("\n####begin, " + dataset + ", model_type==addIter" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    runNoName(datasets=dataset + '/', division='1/', model_type='addIter', is_cuda=is_cuda, arg=myarg)

    # dataset = 'DWY100K/dbp_yg'
    # print("\n####begin, " + dataset + ", model_type==addGCN" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    # runNoName(datasets=dataset + '/', division='1/', model_type='addGCN', is_cuda=is_cuda, arg=myarg)
    # print("\n####begin, " + dataset + ", model_type==addIter" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    # runNoName(datasets=dataset + '/', division='1/', model_type='addIter', is_cuda=is_cuda, arg=myarg)



