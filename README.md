# huggingfaceinference
Simple inference pipelines using hugging transformers library and finetuned tiny models. Will be highly useful on scenarios where we need to optimize storage and memory cost with a compensation in accuracy.

# Installation
```pip install huggingfaceinference```

# Implementation

## TinyGram

A simple grammatical and spelling resolver using huggingface transformers. The Dataset (around 50k) is retrieved from kaggle and corrupted using random substution of letters in words for finetuning. The base model used was _google/t5-efficient-tiny-nl32_ as it is highly compact and efficient(<250 MB).(was modeled on fp32 to reduce drop in efficiency). 

```python

from huggingfaceinference.pipeline import TinyGram

tg = TinyGram()
tg.gramcorrector("What do you think I shold be doing", n=3)

""" Output: ['What do you think I need to be doing',
 'what do you think I should be doing',
 'what do you think I will be doing']"""

# The above pipeline is built by finetuing "google/t5-efficient-tiny-nl32" model on sentences which are corrupted by random noising.
```

## Knowledge Graph

A simple tool to create knowledge graphs in NLP using a pre-trained model modelled on custom dataset created using spaCy library.

```python

from huggingfaceinference.pipeline import KnowledgeGraph

kg = KnowledgeGraph()
kg.get_graph(text)

""" [{'tend': 'O'},
 {'##ul': 'SRC'},
 {'##kar': 'SRC'},
 {'plays': 'SRC'},
 {'for': 'REL'},
 {'india': 'REL'}]"""

# The above pipeline is built by finetuing "google/t5-efficient-tiny-nl32" model on sentences which are corrupted by random noising.
```
