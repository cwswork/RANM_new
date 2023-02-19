# RANM

Source code and datasets for TKDE2022 paper: [***Semi-supervised Entity Alignment via Relation-based Adaptive Neighborhood Matching***]

## Datasets

> Please first download the datasets [here](https://www.jianguoyun.com/p/DY8iIAsQ2t_lCBjK3oUEIAA) and extract them into `datasets/` directory.

Initial datasets WN31-15K is from [OpenEA](https://github:com/nju-websoft/OpenEA).
Initial datasets DBP-15K is from [JAPE](https://github.com/nju-websoft/JAPE).
Initial datasets DWY100K is from [BootEA](https://github.com/nju-websoft/BootEA).

Take the dataset EN_DE(V1) as an example, the folder "pre4" contains:
* kg1_ent_dict: ids for entities in source KG;
* kg2_ent_dict: ids for entities in target KG;
* rel_triples_id: relation triples encoded by ids;
* kgs_num: statistics of the number of entities, relations, attributes, and attribute values;
* entity_embedding.out: the input attribute value feature matrix initialized by word vectors;


## Environment

* Python>=3.7
* pytorch>=1.7.0
* tensorboardX>=2.1.0
* Numpy
* json


## Running

To run RANM model on WN31-15K and DBP-15K, use the following script:
```
python3 align_exc.py
```

> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit (±1%) when running code repeatedly.

> If you have any difficulty or question in running code and reproducing expriment results, please email to cwswork@qq.com.

## Citation

If you use this model or code, please cite it as follows:

Weishan Cai, Wenjun Ma, Lina Wei, and Yuncheng Jiang*. Semi-supervised Entity Alignment via Relation-based Adaptive Neighborhood Matching, IEEE Transactions on Knowledge and Data Engineering(TKDE), Early Access Article. DOI: 10.1109/TKDE.2022.3222811, 2022. (CCF A类期刊)

