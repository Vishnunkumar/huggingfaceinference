# huggingfaceinference
Simple inference usecases using hugging transformers library

# Installation
pip install huggingfaceinference

# Implementation

```python

from huggingfaceinference.pipeline import gramcorrector

gramcorrector("What do you think I shold be doing", n=3)
""" Output: ['What do you think I need to be doing',
 'what do you think I should be doing',
 'what do you think I will be doing']"""

# The above pipeline is built by finetuing "google/t5-efficient-tiny-nl32" model on sentences which are corrupted by random noising.
```
