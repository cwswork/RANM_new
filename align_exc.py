
import os
import time
import torch
from align.model_set import align_union_set
from autil.configUtil import configClass

if __name__ == '__main__':
    #  args
    # EN_DE_15K_V1, EN_FR_15K_V1, fr_en(dbp15) ja_en(dbp15), zh_en(dbp15)
    # 1-5, tt5, tt10, tt15, tt25, tt30
    myconfig = configClass('args_15K.json', datasets='EN_FR_15K_V2/', division='1/')
    #运行设置
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    myconfig.is_cuda = False and torch.cuda.is_available()  # cuda
    myconfig.batch_size = 1024*40
    ###### Model Choice
    model_type = 'addIter' # noGCN, addGCN, addIter
    myconfig.output += 'neigh(0609)_' + model_type + '2/' + myconfig.time_str + '/'
    if model_type == 'noGCN':
        myconfig.isGCN = False  # add GCN, Defalut True
        myconfig.patience = 10
    if model_type == 'addIter':
        myconfig.isIter = True  # add Iter，Defalut False
        myconfig.patience = 20

    #########  Model Definition
    myconfig.set_myprint(os.path.realpath(__file__))
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

