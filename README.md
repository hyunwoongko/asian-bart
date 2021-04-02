# Asian Bart
- `asian-bart` is package of Bart model for Asian languages.
- `asian-bart` supports English, Chinese, Korean, japanese, Total (=ECJK)
- We made `asian-bart` using [mBart](https://arxiv.org/abs/2001.08210) by embedding layer pruning. 
<br><br><br>
  
## Installation
```console
pip install asian-bart
```
<br><br>

## Model specification
- ECJK model
  - vocab size: 57k
  - model size: 413M
  - languages: En, Zh, Ja, Ko
  - architecture: Transformer 12 Encoder + 12 Decoder
  - name: `hyunwoongko/asian-bart-ecjk`
<br><br>  
- English model
  - vocab size: 32k
  - model size: 387M
  - languages: English (`en_XX`)
  - architecture: Transformer 12 Encoder + 12 Decoder
  - name: `hyunwoongko/asian-bart-en`
<br><br>
- Chinese model
  - vocab size: 20k
  - model size: 375M
  - languages: Chinese (`zh_CN`)
  - architecture: Transformer 12 Encoder + 12 Decoder
  - name: `hyunwoongko/asian-bart-zh`
<br><br>
- Japanese model
  - vocab size: 13k
  - model size: 368M
  - languages: Japanese (`ja_XX`)
  - architecture: Transformer 12 Encoder + 12 Decoder
  - name: `hyunwoongko/asian-bart-ja`
 <br><br>
- Korean model
  - vocab size: 8k
  - model size: 363M
  - languages: Korean (`ko_KR`)
  - architecture: Transformer 12 Encoder + 12 Decoder
  - name: `hyunwoongko/asian-bart-ko`
<br><br>
    
## Usage
- The `asian-bart` is made using mbart, so you have to follow mbart's input rules:
  - source: `text` + `</s>` + `lang_code`
  - target: `lang_code` + `text` + `</s>`
- For more details, please check the content of the [mbart paper](https://arxiv.org/abs/2001.08210).
<br><br>
    
### Usage of tokenizer
- tokenization of `(single language, single text)`
```python
>>> from asian_bart import AsianBartTokenizer
>>> tokenizer = AsianBartTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk")
>>> tokenizer.prepare_seq2seq_batch(
...     src_texts="hello.",
...     src_langs="en_XX",
... )
```
```
{
  'input_ids': tensor([[37199, 35816,     2, 57521]]), 
  'attention_mask': tensor([[1, 1, 1, 1]])
}
```
<br>

- batch tokenization of `(single language, mutiple texts)`
```python
>>> from asian_bart import AsianBartTokenizer
>>> tokenizer = AsianBartTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk")
>>> tokenizer.prepare_seq2seq_batch(
...     src_texts=["hello.", "how are you?", "good."],
...     src_langs="en_XX",
... )
```
```
{
  'input_ids': tensor([[37199, 35816,     2, 57521,     1,     1],
                       [38248, 46819, 39446, 36209,     2, 57521],
                       [40010, 39539,     2, 57521,     1,     1]]), 

  'attention_mask': tensor([[1, 1, 1, 1, 0, 0],
                            [1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 0, 0]])
}
```
<br>

- batch tokenization of `(multiple languages, multiple texts)`
```python
>>> from asian_bart import AsianBartTokenizer
>>> tokenizer = AsianBartTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk")
>>> tokenizer.prepare_seq2seq_batch(
...     src_texts=["hello.", "반가워", "你好", "こんにちは"],
...     src_langs=["en_XX", "ko_KR", "zh_CN", "ja_XX"],
... )
```
```
{
  'input_ids': tensor([[37199, 35816, 39539,     2, 57521,     1,     1,     1],
                       [22880, 49591,  3901,     2, 57523,     1,     1,     1],
                       [50356,  7929,     2, 57524,     1,     1,     1,     1],
                       [42990, 19092, 51547, 36821, 33899, 37382,     2, 57522]]), 
 
   'attention_mask': tensor([[1, 1, 1, 1, 1, 0, 0, 0],
                             [1, 1, 1, 1, 1, 0, 0, 0],
                             [1, 1, 1, 1, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 1, 1, 1]])
}
```
<br>

- seq2seq tokenization of `(source text, target text)`
```python
>>> from asian_bart import AsianBartTokenizer
>>> tokenizer = AsianBartTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk")
>>> tokenizer.prepare_seq2seq_batch(
...     src_texts="반가워",
...     src_langs="ko_KR",
...     tgt_texts="hello.",
...     tgt_langs="en_XX",
... )
```
```
{
  'input_ids': tensor([[22880, 49591,  3901,     2, 57523]]), 
  'attention_mask': tensor([[1, 1, 1, 1, 1]]), 
  'labels': tensor([[37199, 35816, 39539,     2, 57521]])
}
```
<br>

- all above batch tokenization settings work the same about target texts
```python
>>> from asian_bart import AsianBartTokenizer
>>> tokenizer = AsianBartTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk")
>>> tokenizer.prepare_seq2seq_batch(
...     src_texts=["hello.", "반가워", "你好", "こんにちは"],
...     src_langs=["en_XX", "ko_KR", "zh_CN", "ja_XX"],
...     tgt_texts=["hello.", "반가워", "你好", "こんにちは"],
...     tgt_langs=["en_XX", "ko_KR", "zh_CN", "ja_XX"],
... )
```
```
{
  'input_ids': tensor([[37199, 35816, 39539,     2, 57521,     1,     1,     1],
                      [22880, 49591,  3901,     2, 57523,     1,     1,     1],
                      [50356,  7929,     2, 57524,     1,     1,     1,     1],
                      [42990, 19092, 51547, 36821, 33899, 37382,     2, 57522]]), 

  'attention_mask': tensor([[1, 1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1]]), 

  'labels': tensor([[37199, 35816, 39539,     2, 57521,     1,     1,     1],
                    [22880, 49591,  3901,     2, 57523,     1,     1,     1],
                    [50356,  7929,     2, 57524,     1,     1,     1,     1],
                    [42990, 19092, 51547, 36821, 33899, 37382,     2, 57522]])
}
```
<br><br>

### Usage of models
- Interfaces of all functions are the same as mbart model on Huggingface transformers.
- Here is an example of using a asian bart model. (ecjk model)
- Other language work the same way. change both model and tokenizer's `from_pretrained`.
    - English only: `from_pretrained("hyunwoongko/asian-bart-en")`
    - Chinese only: `from_pretrained("hyunwoongko/asian-bart-zh")`
    - Japanese only: `from_pretrained("hyunwoongko/asian-bart-ja")`
    - Korean only: `from_pretrained("hyunwoongko/asian-bart-ko")`
```python
# import modules
>>> import torch
>>> from asian_bart import AsianBartTokenizer, AsianBartForConditionalGeneration
>>> from transformers.models.bart.modeling_bart import shift_tokens_right

# create model and tokenizer
>>> tokenizer = AsianBartTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk")
>>> model = AsianBartForConditionalGeneration.from_pretrained("hyunwoongko/asian-bart-ecjk")

# tokenize texts
>>> tokens = tokenizer.prepare_seq2seq_batch(
...     src_texts="Kevin is the <mask> man in the world.",
...     src_langs="en_XX",
...     tgt_texts="Kevin is the most kind man in the world.",
...     tgt_langs="en_XX",                  
... )

>>> input_ids = tokens["input_ids"]
>>> attention_mask = tokens["attention_mask"]
>>> labels = tokens["labels"]
>>> decoder_input_ids = shift_tokens_right(labels, tokenizer.pad_token_id)

# forwarding model for training
>>> output = model(
...     input_ids=input_ids,
...     attention_mask=attention_mask,
...     decoder_input_ids=decoder_input_ids,
... )

# compute loss
>>> lm_logits = outputs[0]
>>> loss_function = torch.nn.CrossEntropyLoss(
...     ignore_index=tokenizer.pad_token_id
... )

>>> loss = loss_function(
...     lm_logits.view(-1, lm_logits.shape[-1]), 
...     labels.view(-1)
... )

# generate text
>>> output = model.generate(
...     input_ids=input_ids,
...     attention_mask=attention_mask,
...     decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"],
... )
```
<br><br>

### Downstream tasks
- You can train various downstream tasks with asian bart.
- All interfaces have the same usage as the Huggingface transformers.
- Supported classes: 
  - `AsianBartTokenizer`
  - `AsianBartModel`
  - `AsianBartForCausalLM`
  - `AsianBartForQuestionAnswering`
  - `AsianBartForConditionalGeneration`
  - `AsianBartForSequenceClassification`
<br><br><br>
    
## License
```
Copyright 2021 Hyunwoong Ko.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
