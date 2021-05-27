<h1 align="center">Icelandic NER  🇮🇸</h1>

## Introduction
This repo consists of pretrained models that were fine-tuned on the MIM-GOLD-NER dataset for the Icelandic language. 
The [MIM-GOLD-NER](http://hdl.handle.net/20.500.12537/42) corpus was developed at [Reykjavik University](https://en.ru.is/) in 2018–2020 that covered eight types of entities:

- Date
- Location
- Miscellaneous 
- Money
- Organization
- Percent
- Person
- Time 

## Dataset Information

|       |   Records |   B-Date |   B-Location |   B-Miscellaneous |   B-Money |   B-Organization |   B-Percent |   B-Person |   B-Time |   I-Date |   I-Location |   I-Miscellaneous |   I-Money |   I-Organization |   I-Percent |   I-Person |   I-Time |
|:------|----------:|---------:|-------------:|------------------:|----------:|-----------------:|------------:|-----------:|---------:|---------:|-------------:|------------------:|----------:|-----------------:|------------:|-----------:|---------:|
| Train |     39988 |     3409 |         5980 |              4351 |       729 |             5754 |         502 |      11719 |      868 |     2112 |          516 |              3036 |       770 |             2382 |          50 |       5478 |      790 |
| Valid |      7063 |      570 |         1034 |               787 |       100 |             1078 |         103 |       2106 |      147 |      409 |           76 |               560 |       104 |              458 |           7 |        998 |      136 |
| Test  |      8299 |      779 |         1319 |               935 |       153 |             1315 |         108 |       2247 |      172 |      483 |          104 |               660 |       167 |              617 |          10 |       1089 |      158 |


## Evaluation

The following tables summarize the scores obtained by pretrained models overall and per each class.

### RoBERTa [IceBERT](https://huggingface.co/mideind/IceBERT)

|     entity    | precision |  recall  | f1-score | support |
|:-------------:|:---------:|:--------:|:--------:|:-------:|
|      Date     |  0.961881 | 0.971759 | 0.966794 |  779.0  |
|    Location   |  0.963047 | 0.968158 | 0.965595 |  1319.0 |
| Miscellaneous |  0.884946 | 0.880214 | 0.882574 |  935.0  |
|     Money     |  0.980132 | 0.967320 | 0.973684 |  153.0  |
|  Organization |  0.924300 | 0.928517 | 0.926404 |  1315.0 |
|    Percent    |  1.000000 | 1.000000 | 1.000000 |  108.0  |
|     Person    |  0.978591 | 0.976413 | 0.977501 |  2247.0 |
|      Time     |  0.965116 | 0.965116 | 0.965116 |  172.0  |
|   micro avg   |  0.951258 | 0.952476 | 0.951866 |  7028.0 |
|   macro avg   |  0.957252 | 0.957187 | 0.957209 |  7028.0 |
|  weighted avg |  0.951237 | 0.952476 | 0.951849 |  7028.0 |


### BERT [mBERT](https://huggingface.co/bert-base-multilingual-cased)

|     entity    | precision |  recall  | f1-score | support |
|:-------------:|:---------:|:--------:|:--------:|:-------:|
|      Date     |  0.969466 | 0.978177 | 0.973802 |  779.0  |
|    Location   |  0.955201 | 0.953753 | 0.954476 |  1319.0 |
| Miscellaneous |  0.867033 | 0.843850 | 0.855285 |  935.0  |
|     Money     |  0.979730 | 0.947712 | 0.963455 |  153.0  |
|  Organization |  0.893939 | 0.897338 | 0.895636 |  1315.0 |
|    Percent    |  1.000000 | 1.000000 | 1.000000 |  108.0  |
|     Person    |  0.963028 | 0.973743 | 0.968356 |  2247.0 |
|      Time     |  0.976879 | 0.982558 | 0.979710 |  172.0  |
|   micro avg   |  0.938158 | 0.938958 | 0.938558 |  7028.0 |
|   macro avg   |  0.950659 | 0.947141 | 0.948840 |  7028.0 |
|  weighted avg |  0.937845 | 0.938958 | 0.938363 |  7028.0 |


### DistilBERT [mdBERT](https://huggingface.co/distilbert-base-multilingual-cased)

|     entity    | precision |  recall  | f1-score | support |
|:-------------:|:---------:|:--------:|:--------:|:-------:|
|      Date     |  0.969309 | 0.973042 | 0.971172 |  779.0  |
|    Location   |  0.941221 | 0.946929 | 0.944067 |  1319.0 |
| Miscellaneous |  0.848283 | 0.819251 | 0.833515 |  935.0  |
|     Money     |  0.928571 | 0.934641 | 0.931596 |  153.0  |
|  Organization |  0.874147 | 0.876806 | 0.875475 |  1315.0 |
|    Percent    |  1.000000 | 1.000000 | 1.000000 |  108.0  |
|     Person    |  0.956674 | 0.972853 | 0.964695 |  2247.0 |
|      Time     |  0.965318 | 0.970930 | 0.968116 |  172.0  |
|   micro avg   |  0.926110 | 0.929141 | 0.927623 |  7028.0 |
|   macro avg   |  0.935441 | 0.936807 | 0.936079 |  7028.0 |
|  weighted avg |  0.925578 | 0.929141 | 0.927301 |  7028.0 |


## How To Use
You use this model with Transformers pipeline for NER.

### Installing requirements

```bash
pip install sentencepiece
pip install transformers
```

### How to predict using pipeline

```python
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification  # for pytorch
from transformers import TFAutoModelForTokenClassification  # for tensorflow
from transformers import pipeline

# model_name_or_path = "m3hrdadfi/icelandic-ner-bert"  # BERT
model_name_or_path = "m3hrdadfi/icelandic-ner-roberta"  # Roberta
# model_name_or_path = "m3hrdadfi/icelandic-ner-distilbert"  # Distilbert

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)  # Pytorch
# model = TFAutoModelForTokenClassification.from_pretrained(model_name_or_path)  # Tensorflow

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Kristin manneskja getur ekki lagt frásagnir af Jesú Kristi á hilluna vegna þess að hún sé búin að lesa þær ."

ner_results = nlp(example)
print(ner_results)
```

## Models

### Hugging Face Model Hub

- [Icelandic NER BERT](https://huggingface.co/m3hrdadfi/icelandic-ner-bert)
- [Icelandic NER RoBERTa](https://huggingface.co/m3hrdadfi/icelandic-ner-roberta)
- [Icelandic NER DistilBERT](https://huggingface.co/m3hrdadfi/icelandic-ner-distilbert)

### Training
All models were trained on a single NVIDIA P100 GPU with following parameters.

**Arguments**
```bash
"task_name": "ner"
"model_name_or_path": model_name_or_path
"train_file": "/content/ner/train.csv"
"validation_file": "/content/ner/valid.csv"
"test_file": "/content/ner/test.csv"
"output_dir": output_dir
"cache_dir": "/content/cache"
"per_device_train_batch_size": 16
"per_device_eval_batch_size": 16
"use_fast_tokenizer": True
"num_train_epochs": 15.0
"do_train": True
"do_eval": True
"do_predict": True
"learning_rate": 2e-5
"evaluation_strategy": "steps"
"logging_steps": 1000
"save_steps": 1000
"save_total_limit": 2
"overwrite_output_dir": True
"fp16": True
"preprocessing_num_workers": 4
```


## Cite
Please cite this repository in publications as the following:

```bibtext
@misc{IcelandicNER,
  author = {Mehrdad Farahani},
  title = {Pre-Trained model for Icelandic NER},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/m3hrdadfi/icelandic-ner}},
}
```


## Questions?
Post a Github issue on the [IcelandicNER Issues](https://github.com/m3hrdadfi/icelandic-ner/issues) repo.
