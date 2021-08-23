# RANM

Source code and datasets for 2021 paper: [***Semi-supervised Cross-lingual Entity Alignment via Relation-based Adaptive Neighborhood Matching***]

## Datasets

> Please first download the datasets and extract them into `datasets/` directory.

Initial datasets WN31-15K and DBP-15K are from [OpenEA](https://github:com/nju-websoft/OpenEA) and [JAPE](https://github.com/nju-websoft/JAPE).

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
python3 exc_plan.py
```

> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit (±1%) when running code repeatedly.

> If you have any difficulty or question in running code and reproducing expriment results, please email to cwswork@qq.com.

## Citation

If you use this model or code, please cite it as follows:

*Weishan Cai, Wenjun Ma, Lina Wei and Yuncheng Jiang. Semi-supervised Cross-lingual Entity Alignment via Relation-based Adaptive Neighborhood Matching. 2021.*
