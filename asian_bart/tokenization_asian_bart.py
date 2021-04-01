import torch
from typing import List, Optional, Dict
from transformers.models.xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer

_all_mbart_models = [
    "hyunwoongko/asian-bart-ecjk",
    "hyunwoongko/asian-bart-en",
    "hyunwoongko/asian-bart-ko",
    "hyunwoongko/asian-bart-zh",
    "hyunwoongko/asian-bart-ja",
]

class AsianBartTokenizer(XLMRobertaTokenizer):

    def __init__(self, *args, tokenizer_file=None, src_lang=None, tgt_lang=None, **kwargs):
        super().__init__(*args, tokenizer_file=tokenizer_file, src_lang=src_lang, tgt_lang=tgt_lang, **kwargs)
        self.vocab_files_names = {"vocab_file": self.vocab_file}
        self.max_model_input_sizes = {m: 1024 for m in _all_mbart_models}
        self.sp_model_size = len(self.sp_model)
        self.pretrained_vocab_files_map = {
            "vocab_file": {
                "hyunwoongko/asian-bart-ecjk": "https://huggingface.co/hyunwoongko/asian-bart-ecjk/resolve/main/sentencepiece.bpe.model",
                "hyunwoongko/asian-bart-en": "https://huggingface.co/hyunwoongko/asian-bart-en/resolve/main/sentencepiece.bpe.model",
                "hyunwoongko/asian-bart-ja": "https://huggingface.co/hyunwoongko/asian-bart-ja/resolve/main/sentencepiece.bpe.model",
                "hyunwoongko/asian-bart-ko": "https://huggingface.co/hyunwoongko/asian-bart-ko/resolve/main/sentencepiece.bpe.model",
                "hyunwoongko/asian-bart-zh": "https://huggingface.co/hyunwoongko/asian-bart-zh/resolve/main/sentencepiece.bpe.model",
            }
        }

        if "asian-bart-ecjk" in kwargs["name_or_path"]:
            langs = ["en_XX", "ja_XX", "ko_KR", "zh_CN"]
        elif "asian-bart-en" in kwargs["name_or_path"]:
            langs = ["en_XX"]
        elif "asian-bart-ja" in kwargs["name_or_path"]:
            langs = ["ja_XX"]
        elif "asian-bart-ko" in kwargs["name_or_path"]:
            langs = ["ko_KR"]
        elif "asian-bart-zh" in kwargs["name_or_path"]:
            langs = ["zh_CN"]
        else:
            raise Exception("wrong pretrained model name.")

        self.prefix_tokens: List[int] = []
        self.suffix_tokens: List[int] = []
        self.lang_code_to_id = {
            code: self.sp_model_size + i + self.fairseq_offset
            for i, code in enumerate(langs)
        }

        self.id_to_lang_code = {v: k for k, v in self.lang_code_to_id.items()}
        self.fairseq_tokens_to_ids["<mask>"] = (len(self.sp_model) +
                                                len(self.lang_code_to_id) +
                                                self.fairseq_offset)

        self.fairseq_tokens_to_ids.update(self.lang_code_to_id)
        self.fairseq_ids_to_tokens = {
            v: k for k, v in self.fairseq_tokens_to_ids.items()
        }
        self.add_special_tokens({"bos_token": "<s>"})
        self.add_special_tokens({"eos_token": "</s>"})
        self.add_special_tokens({"pad_token": "<pad>"})
        self.add_special_tokens({"unk_token": "<unk>"})
        self.add_special_tokens({"mask_token": "<mask>"})

    def set_src_lang_special_tokens(self, src_lang) -> None:
        self.cur_lang_code = self.lang_code_to_id[src_lang]
        self.prefix_tokens = []
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        self.cur_lang_code = self.lang_code_to_id[lang]
        self.prefix_tokens = []
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_langs: List[str],
        tgt_texts: List[str] = None,
        tgt_langs: List[str] = None,
        max_len: int = 1024,
    ) -> Dict:

        if isinstance(src_texts, str):
            src_texts = [src_texts]

        if isinstance(src_langs, str):
            src_langs = [src_langs] * len(src_texts)

        if isinstance(tgt_texts, str):
            tgt_texts = [tgt_texts]

        if isinstance(tgt_langs, str):
            tgt_langs = [tgt_langs] * len(tgt_texts)

        assert len(src_texts) == len(
            src_langs
        ), "src_texts list and src_langs list must have same length"

        src_tokens = self(
            src_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=max_len - 2,
            # result + <eos> + <lang_id>
        )

        src_tokens = self.add_language_tokens(src_tokens, src_langs)

        if not tgt_texts:
            return {
                "input_ids": src_tokens["input_ids"],
                "attention_mask": src_tokens["attention_mask"],
            }

        assert len(tgt_texts) == len(
            tgt_langs
        ), "tgt_texts list and tgt_langs list must have same length"

        tgt_tokens = self(
            tgt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=max_len - 2,
            # result + <eos> + <lang_id>
        )

        tgt_tokens = self.add_language_tokens(tgt_tokens, tgt_langs)

        return {
            "input_ids": src_tokens["input_ids"],
            "attention_mask": src_tokens["attention_mask"],
            "labels": tgt_tokens["input_ids"],
        }

    def add_language_tokens(self, tokens, langs):
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        token_added_ids, token_added_masks = [], []

        for input_id, atn_mask, lang in zip(input_ids, attention_mask, langs):
            maximum_idx = [
                i for i, val in enumerate(input_id) if val != self.pad_token_id
            ]

            if len(maximum_idx) == 0:
                idx_to_add = 0
            else:
                idx_to_add = max(maximum_idx) + 1

            eos = self.eos_token_id
            lang = self.lang_code_to_id[lang]

            special_tokens = torch.tensor([eos, lang], requires_grad=False)
            input_id = torch.cat(
                [input_id[:idx_to_add], special_tokens,
                 input_id[idx_to_add:]]).long()

            additional_attention_mask = torch.tensor([1, 1],
                                                     requires_grad=False)
            atn_mask = torch.cat([
                atn_mask[:idx_to_add],
                additional_attention_mask,
                atn_mask[idx_to_add:],
            ]).long()

            token_added_ids.append(input_id.unsqueeze(0))
            token_added_masks.append(atn_mask.unsqueeze(0))

        tokens["input_ids"] = torch.cat(token_added_ids, dim=0)
        tokens["attention_mask"] = torch.cat(token_added_masks, dim=0)
        return tokens

    def build_inputs_with_special_tokens(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None) -> List[int]:

        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(
                map(
                    lambda x: 1
                    if x in [self.sep_token_id, self.cls_token_id] else 0,
                    token_ids_0,
                ))
        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1] * len(self.suffix_tokens)
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        return (prefix_ones + ([0] * len(token_ids_0)) +
                ([0] * len(token_ids_1)) + suffix_ones)
