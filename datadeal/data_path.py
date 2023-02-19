import math
import os
import time
import random
import numpy as np

from autil import fileUtil

def savePath(filename, path_list):
    with open(datasetPath + filename, 'w', encoding='utf-8') as fw:
        for line in path_list:
            fw.write('{}\n'.format('\t'.join('%s' %id for id in line)))


def set_rPath_1(datasetPath, rel_triple):  #     r_head = {}
    for (h, r, t) in rel_triple:
        r_head[h] = r_head.get(h, list()) + [(r, t)]

    path_list = []
    for (h, r, t) in rel_triple:
        if t in r_head.keys():
            for (r2, t2) in r_head[t]:  # (h, r, t) + (t, r2, t2)
                path_list.append((h, r, t, r2, t2))  # t == hh

    savePath(datasetPath +'pre2/r_path-2.txt', path_list)

    resetPath(datasetPath, 2)


def set_rPath_multi(datasetPath, rel_triple, pathlen):

    r_head = {}
    for (h, r, t) in rel_triple:
        r_head[h] = r_head.get(h, list()) + [(r, t)]

    if pathlen == 2:
        old_path = rel_triple
    else:
        # 读取路径
        old_path = []
        pathFile = datasetPath +'pre2/r_path-' + str(pathlen-1) + '.txt'
        with open(pathFile, encoding='utf-8', mode='r') as f:
            for line in f:
                th = line[:-1].split('\t')
                th = [int(i) for i in th]
                old_path.append(th)

    new_path = []
    for pp in old_path:
        t = pp[-1]  #   # (h, r, ... , t) + (t, r3, t3)
        if t in r_head.keys():
            for (r3, t3) in r_head[t]:
                new_path.append(pp + [r3, t3])


    savePath(datasetPath + 'pre2/r2_path-' + str(pathlen) + '.txt', new_path)
    resetPath(datasetPath, pathlen)


def resetPath(datasetPath, pathlen):

    path_file = datasetPath + 'pre2/r2_path-' + str(pathlen) + '.txt'
    path_triple = fileUtil.load_triples_list(path_file)  #
    rr_dict = {}
    for pp in path_triple:
        if len(pp) == 5:
            rr = (pp[1], pp[3])
        elif len(pp) == 7:
            rr = (pp[1], pp[3], pp[5])
        elif len(pp) == 9:
            rr = (pp[1], pp[3], pp[5], pp[7])

        if rr not in rr_dict.keys():
            rr_dict[rr] = []

        rr_dict[rr].append(tuple(pp))

    #
    rr_list = sorted(rr_dict.items(), key=lambda x: len(x[1]), reverse=True)
    with open(datasetPath + 'pre2/r2_path_list-' + str(pathlen) + '.txt', 'w', encoding='utf-8') as fw:
        for path in rr_list:
            rrr = path[0].__str__()  # rr
            pp_str = path[1].__str__()  #
            entity = set()
            for pp in path[1]:
                if len(pp) == 5:
                    entity |= set([pp[0], pp[2], pp[4]])
                elif len(pp) == 7:
                    entity |= set([pp[0], pp[2], pp[4], pp[6]])
                elif len(pp) == 9:
                    entity |= set([pp[0], pp[2], pp[4], pp[6], pp[8]])

            fw.write('{}\t{}\t{}\t{}\n'.format(rrr, str(len(path[1])), str(len(entity)), pp_str))  # (r1,r2)    len


if __name__ == '__main__':
    print("start==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    # EN_DE_15K_V1、EN_FR_15K_V1
    datasetPath = '../../0315datasets/EN_DE_15K_V1/'
    dataset_division = '721_5fold/1/'
    seed = 72
    ordered = True
    print(datasetPath)
    random.seed(seed)
    np.random.seed(seed)

    kgs_num_dict = fileUtil.load_dict(datasetPath + 'pre/kgs_num', read_kv='kv', sep=':')
    kg_E = int(kgs_num_dict['KG_E'])  # KG_E
    kg_R = int(kgs_num_dict['KG_R'])  # KG_R
    rel_triple = fileUtil.load_triples_list(datasetPath + 'pre/rel_triples_id')

    set_rPath_multi(datasetPath, rel_triple, 2)

    print("end==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
