import json
import math
import os
import time
import random
import numpy as np

from autil import fileUtil
from datadeal import pre_embeds


def set_relation2id_byName(datasetPath, ordered=True):
    kg1_rel_triples, kg1_entities, kg1_relations = fileUtil.read_relation_triples(
        datasetPath + 'rel_triples_1')  # KG1 rel_triples_1
    kg2_rel_triples, kg2_entities, kg2_relations = fileUtil.read_relation_triples(
        datasetPath + 'rel_triples_2')  # kg2

    # entity #
    kg1_ent_dict, kg2_ent_dict = fileUtil.gen_mapping_id(kg1_rel_triples, kg1_entities, kg2_rel_triples,
                                                         kg2_entities, ordered=ordered)
    ent_dict = dict(kg1_ent_dict, **kg2_ent_dict)
    assert len(kg1_ent_dict) + len(kg2_ent_dict) == len(ent_dict)

    fileUtil.save_dict2txt(datasetPath + 'pre4/ent_dict', ent_dict, save_kv='vk')  # ent_dict{name:id)
    fileUtil.save_dict2txt(datasetPath + 'pre4/kg1_ent_dict', kg1_ent_dict, save_kv='vk')
    fileUtil.save_dict2txt(datasetPath + 'pre4/kg2_ent_dict', kg2_ent_dict, save_kv='vk')

    # relation
    kg1_rel_dict, kg2_rel_dict = fileUtil.gen_mapping_id(kg1_rel_triples, kg1_relations, kg2_rel_triples,
                                                         kg2_relations, ordered=ordered)
    fileUtil.save_dict2txt(datasetPath + 'pre4/kg1_rel_dict', kg1_rel_dict, save_kv='vk')   # ent_dict{name:id)
    fileUtil.save_dict2txt(datasetPath + 'pre4/kg2_rel_dict', kg2_rel_dict, save_kv='vk')

    #  rel_triples_id
    kg1_rel_triples_id = fileUtil.relation_triple2ids(kg1_rel_triples, kg1_ent_dict, kg1_rel_dict)  # (h, r, t)
    kg2_rel_triples_id = fileUtil.relation_triple2ids(kg2_rel_triples, kg2_ent_dict, kg2_rel_dict)
    rel_triples_id = kg1_rel_triples_id + kg2_rel_triples_id
    fileUtil.save_triple2txt(datasetPath + 'pre4/rel_triples_id', rel_triples_id)

    print("Num of KG1 entitys:", len(kg1_ent_dict))
    print("Num of KG2 entitys:", len(kg2_ent_dict))
    print("Num of KG1 relations:", len(kg1_rel_dict))
    print("Num of KG2 relations:", len(kg2_rel_dict))
    print("Num of KGs rel triples:", len(rel_triples_id))

    with open(datasetPath + 'pre4/kgs_num', 'w') as ff:
        ff.write('KG_E\t' + str(len(kg1_ent_dict) + len(kg2_ent_dict)) + '\n')
        ff.write('KG_R\t' + str(len(kg1_rel_dict) + len(kg2_rel_dict)) + '\n')


def get_entity_embed(datasetPath):
    # entity embedding
    if 'D_W--' in datasetPath:
        kg_ent_list = fileUtil.load_ids2list(datasetPath + 'pre4/ent_dict_new')
    else:
        kg_ent_list = fileUtil.load_ids2list(datasetPath + 'pre4/ent_dict')

    entity_name_list, entity_embedding = pre_embeds.get_entity_embed(kg_ent_list)
    fileUtil.save_list2txt(datasetPath + 'pre4/replace_entity_name.txt', entity_name_list)
    fileUtil.savepickle(datasetPath + 'pre4/entity_embedding.out', entity_embedding)  # ndarray(30000,300) ,按两个kg的e编号


### train_links... ###
def read_links(file_path):
    '''
     read read_links
    :param file_path: such as 'datasets\D_W_15K_V1\721_5fold\1\train_links'
    :return: train_links
    '''
    print("read links:", file_path)
    links = list()
    refs = list()
    reft = list()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        e1 = params[0].strip()
        e2 = params[1].strip()
        refs.append(e1)
        reft.append(e2)
        links.append((e1, e2))
    assert len(refs) == len(reft)
    return links


def set_links_ID(datasetPath, dataset_division):
    kg1_entity2index = fileUtil.load_ids2dict(datasetPath + 'pre4/kg1_ent_dict', read_kv='vk')
    kg2_entity2index = fileUtil.load_ids2dict(datasetPath + 'pre4/kg2_ent_dict', read_kv='vk')

    train_links_ID = fileUtil.get_links_ids(datasetPath + dataset_division + 'train_links',
                                              kg1_entity2index, kg2_entity2index)
    test_links_ID = fileUtil.get_links_ids(datasetPath + dataset_division + 'test_links',
                                             kg1_entity2index, kg2_entity2index)
    valid_links_ID = fileUtil.get_links_ids(datasetPath + dataset_division + 'valid_links',
                                              kg1_entity2index, kg2_entity2index)
    print('save files...train_links...')
    fileUtil.save_list2txt(datasetPath + dataset_division + 'train_links_id', train_links_ID)
    fileUtil.save_list2txt(datasetPath + dataset_division + 'valid_links_id', valid_links_ID)
    fileUtil.save_list2txt(datasetPath + dataset_division + 'test_links_id', test_links_ID)


# 重置Rel 的id文件
def reset_reldict(datasetPath):
    kg1_rel_dict = dict()
    with open(datasetPath + 'pre4/kg1_rel_dict.txt', encoding='utf-8') as f:
        for line in f:
            line = line[:-1]
            i = line.index(':')
            kg1_rel_dict[line[i+1: ]] = int(line[0:i])  # (name:eid)

    kg2_rel_dict = dict()
    with open(datasetPath + 'kg2_rel_dict.txt', encoding='utf-8') as f:
        for line in f:
            line = line[:-1]
            i = line.index(':')
            kg2_rel_dict[line[i+1: ]] = int(line[0:i])  # (name:eid)

    fileUtil.save_dict2txt(datasetPath + 'kg1_rel_dict', kg1_rel_dict, save_kv='vk')   # ent_dict{name:id)
    fileUtil.save_dict2txt(datasetPath + 'kg2_rel_dict', kg2_rel_dict, save_kv='vk')


if __name__ == '__main__':
    print("start==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    seed = 72
    random.seed(seed)
    np.random.seed(seed)

    # dataset: EN_DE_15K_V1、EN_FR_15K_V1    ja_en(dbp15),fr_en(dbp15),zh_en(dbp15)
    # DWY100K/dbp_wd  DWY100K/dbp_yg
    #datasetPath = '../../0315datasets/DWY100K/dbp_wd/'
    datasetPath = '../../0315datasets/EN_DE_15K_V1/'
    dataset_division = '721_5fold/1/'
    print(datasetPath)
    if not os.path.exists(datasetPath + 'pre4/'):
        print('dir not exists')
        os.makedirs(datasetPath + 'pre4/')

    # 从文件夹读取kg文件，生成实体，属性，关系，属性值的ID
    #set_relation2id_byName(datasetPath, ordered=True)
    #set_entity_links_dbp(datasetPath)

    # 获得实体名，属性，属性值嵌入
    # get_entity_embed(datasetPath)

    # 重新生成训练集，验证集，测试集的ID文件
    #set_links_file(datasetPath, dataset_division)
    # datasetPath = '../../0315datasets/EN_DE_15K_V2/'
    # set_links_ID(datasetPath, '721_5fold/2/')

    # 获取所有实体对的ID
    #set_entity_links(datasetPath)
    print("end==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
