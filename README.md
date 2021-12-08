# PURE: Entity and Relation Extraction from Text
这个repository包含（PyTorch）代码和PURE（the **P**rinceton **U**niversity **R**elation **E**xtraction系统）的预训练模型，由论文: [A Frustratingly Easy Approach for Entity and Relation Extraction](https://arxiv.org/pdf/2010.12812.pdf).

## Quick links
* [Overview](#Overview)
* [Setup](#Setup)
  * [Install dependencies](#Install-dependencies)
  * [Data preprocessing](#Download-and-preprocess-the-datasets)
* [Quick Start](#Quick-start)
* [Entity Model](#Entity-Model)
  * [Input data format](#Input-data-format-for-the-entity-model)
  * [Train/evaluate the entity model](#Train/evaluate-the-entity-model)
* [Relation Model](#Relation-Model)
  * [Input data format](#Input-data-format-for-the-relation-model)
  * [Train/evaluate the relation model](#Train/evaluate-the-relation-model)
  * [Approximation relation model](#Approximation-relation-model)
* [Pre-trained Models](#Pre-trained-Models)
  * [Pre-trained models for ACE05](#Pre-trained-models-for-ACE05)
  * [Pre-trained models for SciERC](#Pre-trained-models-for-SciERC)
* [Bugs or Questions?](#Bugs-or-questions)
* [Citation](#Citation)

## 概览
![](./figs/overview.png)
在这项工作中，我们提出了一种简单的实体和关系抽取方法。我们的方法包含三个部分。

1. 实体模型将一段文本作为输入，并一次性预测所有实体。
2. 关系模型通过插入类型化的实体标记来独立考虑每一对实体，并预测每一对实体的关系类型。
3. 近似关系模型支持批次计算，这使得关系模型的推理更加高效。

Please find more details of this work in our [paper](https://arxiv.org/pdf/2010.12812.pdf).

## Setup

### Install dependencies
请使用以下命令安装所有的依赖包。
```
pip install -r requirements.txt
```

### Download and preprocess the datasets
我们的实验是基于三个数据集的。ACE04, ACE05, 和SciERC。请在下面找到链接和预处理方法。
* ACE04/ACE05：我们使用来自[DyGIE repo]（https://github.com/luanyi/DyGIE/tree/master/preprocessing）的预处理代码。请按照说明对ACE05和ACE04数据集进行预处理。
* SciERC：预处理的SciERC数据集可以在他们的项目[网站]（http://nlp.cs.washington.edu/sciIE/）中下载。

## Quick Start
以下命令可用于下载预处理的SciERC数据集，并在SciERC上运行我们预训练的模型。

```bash
# 下载SciERC 数据集
wget http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz
mkdir scierc_data; tar -xf sciERC_processed.tar.gz -C scierc_data; rm -f sciERC_processed.tar.gz
scierc_dataset=scierc_data/processed_data/json/

# 下载预训练好的模型（单句）。
mkdir scierc_models; cd scierc_models

# 下载预训练好的实体模型
wget https://nlp.cs.princeton.edu/projects/pure/scierc_models/ent-scib-ctx0.zip
unzip ent-scib-ctx0.zip; rm -f ent-scib-ctx0.zip
scierc_ent_model=scierc_models/ent-scib-ctx0/

# 下载预训练好的完整关系模型
wget https://nlp.cs.princeton.edu/projects/pure/scierc_models/rel-scib-ctx0.zip
unzip rel-scib-ctx0.zip; rm -f rel-scib-ctx0.zip
scierc_rel_model=scierc_models/rel-scib-ctx0/

# 下载预训练好的近似关系模型
wget https://nlp.cs.princeton.edu/projects/pure/scierc_models/rel_approx-scib-ctx0.zip
unzip rel_approx-scib-ctx0.zip; rm -f rel_approx-scib-ctx0.zip
scierc_rel_model_approx=scierc_models/rel_approx-scib-ctx0/

cd ..

# 运行预训练好的实体模型，结果将被存储在 ${scierc_ent_model}/ent_pred_test.json
python run_entity.py 
    --do_eval --eval_test 
    --context_window 0 
    --task scierc 
    --data_dir scierc_data/processed_data/json/ 
    --model allenai/scibert_scivocab_uncased 
    --output_dir scierc_models/ent-scib-ctx0/

# 运行预训练好的完整关系模型
python run_relation.py \
  --task scierc \
  --do_eval --eval_test \
  --model allenai/scibert_scivocab_uncased \
  --do_lower_case \
  --context_window 0\
  --max_seq_length 128 \
  --entity_output_dir ${scierc_ent_model} \
  --output_dir ${scierc_rel_model}
  
# 输出端到端的评估结果
python run_eval.py --prediction_file scierc_models/rel-scib-ctx0/predictions.json

# 运行预训练好的近似关系模型（有批次计算）。
python run_relation_approx.py \
  --task scierc \
  --do_eval --eval_test \
  --model allenai/scibert_scivocab_uncased \
  --do_lower_case \
  --context_window 0\
  --max_seq_length 250 \
  --entity_output_dir ${scierc_ent_model} \
  --output_dir ${scierc_rel_model_approx} \
  --batch_computation

# 输出端到端的评估结果
python run_eval.py --prediction_file ${scierc_rel_model_approx}/predictions.json
```

## Entity Model

### 实体模型的输入数据格式

实体模型的输入数据格式是JSONL。输入文件的每一行都包含一个文件，格式如下。

```
{
  # document ID (please make sure doc_key can be used to identify a certain document)
  "doc_key": "CNN_ENG_20030306_083604.6",

  # sentences in the document, each sentence is a list of tokens
  "sentences": [
    [...],
    [...],
    ["tens", "of", "thousands", "of", "college", ...],
    ...
  ],

  # entities (boundaries and entity type) in each sentence
  "ner": [
    [...],
    [...],
    [[26, 26, "LOC"], [14, 14, "PER"], ...], #the boundary positions are indexed in the document level
    ...,
  ],

  # relations (two spans and relation type) in each sentence
  "relations": [
    [...],
    [...],
    [[14, 14, 10, 10, "ORG-AFF"], [14, 14, 12, 13, "ORG-AFF"], ...],
    ...
  ]
}
```

### Train/evaluate the entity model
你可以使用`run_entity.py`和`--do_train`来训练一个实体模型，使用`--do_eval`来评估一个实体模型。
一个训练命令模板如下： 

```bash
python run_entity.py \
    --do_train --do_eval [--eval_test] \
    --learning_rate=1e-5 --task_learning_rate=5e-4 \
    --train_batch_size=16 \
    --context_window {0 | 100 | 300} \
    --task {ace05 | ace04 | scierc} \
    --data_dir {directory of preprocessed dataset} \
    --model {bert-base-uncased | albert-xxlarge-v1 | allenai/scibert_scivocab_uncased} \
    --output_dir {directory of output files}
```
Arguments:
* `--learning_rate`: BERT编码器参数的学习率。
* `--task_learning_rate`: 特定任务参数的学习率，即编码器之后的分类器头。
* `--context_window`: 模型中使用的上下文窗口大小。`0`意味着不使用上下文。在我们的跨句子实体实验中，我们对BERT模型和SciBERT模型使用`--context_window 300`，对ALBERT模型使用`--context_window 100`。
* `--model`:基础transformer模型。我们对ACE04/ACE05使用`bert-base-uncased`和`albert-xxlarge-v1`，对SciERC使用`allenai/scibert_scivocab_uncased`。
* `--eval_test`: 是否在测试集上进行评估。

实体模型的预测结果将被保存为一个文件（`ent_pred_dev.json`），在`output_dir`目录下。如果你设置了`--eval_test`，预测结果（`ent_pred_test.json`）将在测试集上。实体模型的预测文件将是关系模型的输入文件。

## Relation Model
### 关系模型的输入数据格式
关系模型的输入数据格式与实体模型的基本相同，只是多了一个".predicted_ner "文件，用于存储实体模型的预测结果。

```bash
{
  "doc_key": "CNN_ENG_20030306_083604.6",
  "sentences": [...],
  "ner": [...],
  "relations": [...],
  "predicted_ner": [
    [...],
    [...],
    [[26, 26, "LOC"], [14, 15, "PER"], ...],
    ...
  ]
}
```

### Train/evaluate the relation model:
你可以使用`run_relation.py`和`--do_train`来训练一个关系模型，使用`--do_eval`来评估一个关系模型。一个训练命令模板如下。

```bash
python run_relation.py \
  --task {ace05 | ace04 | scierc} \
  --do_train --train_file {path to the training json file of the dataset} \
  --do_eval [--eval_test] [--eval_with_gold] \
  --model {bert-base-uncased | albert-xxlarge-v1 | allenai/scibert_scivocab_uncased} \
  --do_lower_case \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --context_window {0 | 100} \
  --max_seq_length {128 | 228} \
  --entity_output_dir {path to output files of the entity model} \
  --output_dir {directory of output files}
```
Aruguments:
* `--eval_with_gold`: whether evaluate the model with the gold entities provided.
* `--entity_output_dir`: the output directory of the entity model. The prediction files (`ent_pred_dev.json` or `ent_pred_test.json`) of the entity model should be in this directory.

预测结果将存储在`output_dir`文件夹下的`predictions.json`文件中，其格式与实体模型的输出文件几乎相同，只是每个文档多了一个字段`"predicted_relations"`。

你可以运行评估脚本来输出预测的端到端性能（`Ent`，`Rel`，和`Rel+`）。

```bash
python run_eval.py --prediction_file {path to output_dir}/predictions.json
```

### Approximation relation model
你可以使用以下命令来训练一个近似模型。
```bash
python run_relation_approx.py \
 --task {ace05 | ace04 | scierc} \
 --do_train --train_file {path to the training json file of the dataset} \
 --do_eval [--eval_with_gold] \
 --model {bert-base-uncased | allenai/scibert_scivocab_uncased} \
 --do_lower_case \
 --train_batch_size 32 \
 --eval_batch_size 32 \
 --learning_rate 2e-5 \
 --num_train_epochs 10 \
 --context_window {0 | 100} \
 --max_seq_length {128 | 228} \
 --entity_output_dir {path to output files of the entity model} \
 --output_dir {directory of output files}
```

一旦你有了训练好的近似模型，你可以在推理过程中用`--batch_computation`启用高效的批次计算。
```bash
python run_relation_approx.py \
 --task {ace05 | ace04 | scierc} \
 --do_eval [--eval_test] [--eval_with_gold] \
 --model {bert-base-uncased | allenai/scibert_scivocab_uncased} \
 --do_lower_case \
 --eval_batch_size 32 \
 --context_window {0 | 100} \
 --max_seq_length 250 \
 --entity_output_dir {path to output files of the entity model} \
 --output_dir {directory of output files} \
 --batch_computation
```
*注意*：目前的代码不支持基于ALBERT的近似模型。

## Pre-trained Models
我们发布了针对ACE05和SciERC数据集的预训练的实体模型和关系模型。
*注意*：预训练模型的性能可能与论文中的报告数字略有不同，因为我们报告的是基于多次运行的平均数字。


### Pre-trained models for ACE05
**Entity models**:
* [BERT (single, W=0)](https://nlp.cs.princeton.edu/projects/pure/ace05_models/ent-bert-ctx0.zip) (388M): Single-sentence entity model based on `bert-base-uncased`
* [ALBERT (single, W=0)](https://nlp.cs.princeton.edu/projects/pure/ace05_models/ent-alb-ctx0.zip) (793M): Single-sentence entity model based on `albert-xxlarge-v1`
* [BERT (cross, W=300)](https://nlp.cs.princeton.edu/projects/pure/ace05_models/ent-bert-ctx300.zip) (388M): Cross-sentence entity model based on `bert-base-uncased`
* [ALBERT (cross, W=100)](https://nlp.cs.princeton.edu/projects/pure/ace05_models/ent-alb-ctx100.zip) (793M): Cross-sentence entity model based on `albert-xxlarge-v1`

**Relation models**:
* [BERT (single, W=0)](https://nlp.cs.princeton.edu/projects/pure/ace05_models/rel-bert-ctx0.zip) (387M): Single-sentence relation model based on `bert-base-uncased`
* [BERT-approx (single, W=0)](https://nlp.cs.princeton.edu/projects/pure/ace05_models/rel_approx-bert-ctx0.zip) (387M): Single-sentence approximation relation model based on `bert-base-uncased`
* [ALBERT (single, W=0)](https://nlp.cs.princeton.edu/projects/pure/ace05_models/rel-alb-ctx0.zip) (789M): Single-sentence relation model based on `albert-xxlarge-v1`
* [BERT (cross, W=100)](https://nlp.cs.princeton.edu/projects/pure/ace05_models/rel-bert-ctx100.zip) (387M): Cross-sentence relation model based on `bert-base-uncased`
* [BERT-approx (cross, W=100)](https://nlp.cs.princeton.edu/projects/pure/ace05_models/rel_approx-bert-ctx100.zip) (387M): Crosss-sentence approximation relation model based on `bert-base-uncased`
* [ALBERT (cross, W=100)](https://nlp.cs.princeton.edu/projects/pure/ace05_models/rel-alb-ctx100.zip) (789M): Cross-sentence relation model based on `albert-xxlarge-v1`

**Performance of pretrained models on ACE05 test set**:
* BERT (single)
```
NER - P: 0.890260, R: 0.882944, F1: 0.886587
REL - P: 0.689624, R: 0.652476, F1: 0.670536
REL (strict) - P: 0.664830, R: 0.629018, F1: 0.646429
```
* BERT-approx (single)
```
NER - P: 0.890260, R: 0.882944, F1: 0.886587
REL - P: 0.678899, R: 0.642919, F1: 0.660419
REL (strict) - P: 0.651376, R: 0.616855, F1: 0.633646
```
* ALBERT (single)
```
NER - P: 0.900237, R: 0.901388, F1: 0.900812
REL - P: 0.739901, R: 0.652476, F1: 0.693444
REL (strict) - P: 0.698522, R: 0.615986, F1: 0.654663
```
* BERT (cross)
```
NER - P: 0.902111, R: 0.905405, F1: 0.903755
REL - P: 0.701950, R: 0.656820, F1: 0.678636
REL (strict) - P: 0.668524, R: 0.625543, F1: 0.646320
```
* BERT-approx (cross)
```
NER - P: 0.902111, R: 0.905405, F1: 0.903755
REL - P: 0.684448, R: 0.657689, F1: 0.670802
REL (strict) - P: 0.659132, R: 0.633362, F1: 0.645990
```
* ALBERT (cross)
```
NER - P: 0.911111, R: 0.905953, F1: 0.908525
REL - P: 0.748521, R: 0.659427, F1: 0.701155
REL (strict) - P: 0.723866, R: 0.637706, F1: 0.678060
```

### Pre-trained models for SciERC
**Entity models**:
* [SciBERT (single, W=0)](https://nlp.cs.princeton.edu/projects/pure/scierc_models/ent-scib-ctx0.zip) (391M): Single-sentence entity model based on `allenai/scibert_scivocab_uncased`
* [SciBERT (cross, W=300)](https://nlp.cs.princeton.edu/projects/pure/scierc_models/ent-scib-ctx300.zip) (391M): Cross-sentence entity model based on `allenai/scibert_scivocab_uncased`

**Relation models**:
* [SciBERT (single, W=0)](https://nlp.cs.princeton.edu/projects/pure/scierc_models/rel-scib-ctx0.zip) (390M): Single-sentence relation model based on `allenai/scibert_scivocab_uncased`
* [SciBERT-approx (single, W=0)](https://nlp.cs.princeton.edu/projects/pure/scierc_models/rel_approx-scib-ctx0.zip) (390M): Single-sentence approximation relation model based on `allenai/scibert_scivocab_uncased`
* [SciBERT (cross, W=100)](https://nlp.cs.princeton.edu/projects/pure/scierc_models/rel-scib-ctx100.zip) (390M): Cross-sentence relation model based on `allenai/scibert_scivocab_uncased`
* [SciBERT-approx (cross, W=100)](https://nlp.cs.princeton.edu/projects/pure/scierc_models/rel_approx-scib-ctx100.zip) (390M): Cross-sentence approximation relation model based on `allenai/scibert_scivocab_uncased`

**Performance of pretrained models on SciERC test set**:
* SciBERT (single)
```
NER - P: 0.667857, R: 0.665875, F1: 0.666865
REL - P: 0.491614, R: 0.481520, F1: 0.486515
REL (strict) - P: 0.360587, R: 0.353183, F1: 0.356846
```
* SciBERT-approx (single)
```
NER - P: 0.667857, R: 0.665875, F1: 0.666865
REL - P: 0.500000, R: 0.453799, F1: 0.475780
REL (strict) - P: 0.376697, R: 0.341889, F1: 0.358450
```
* SciBERT (cross)
```
NER - P: 0.676223, R: 0.713947, F1: 0.694573
REL - P: 0.494797, R: 0.536961, F1: 0.515017
REL (strict) - P: 0.362346, R: 0.393224, F1: 0.377154
```
* SciBERT-approx (cross)
```
NER - P: 0.676223, R: 0.713947, F1: 0.694573
REL - P: 0.483366, R: 0.507187, F1: 0.494990
REL (strict) - P: 0.356164, R: 0.373717, F1: 0.364729
```

## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Zexuan Zhong `(zzhong@cs.princeton.edu)`. If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation
If you use our code in your research, please cite our work:
```bibtex
@inproceedings{zhong2021frustratingly,
   title={A Frustratingly Easy Approach for Entity and Relation Extraction},
   author={Zhong, Zexuan and Chen, Danqi},
   booktitle={North American Association for Computational Linguistics (NAACL)},
   year={2021}
}
```
