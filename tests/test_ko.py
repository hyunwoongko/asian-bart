from unittest import TestCase
from asian_bart import AsianBartTokenizer, AsianBartForConditionalGeneration


class TestKo(TestCase):

    def test_tokenizer(self):
        tokenizer = AsianBartTokenizer.from_pretrained(
            "hyunwoongko/asian-bart-ko")

        tokens = tokenizer.prepare_seq2seq_batch(
            src_texts="안녕하세요.",
            src_langs="ko_KR",
            tgt_texts="반갑습니다.",
            tgt_langs="ko_KR",
        )

        print(tokens)

    def test_model(self):
        tokenizer = AsianBartTokenizer.from_pretrained(
            "hyunwoongko/asian-bart-ko")

        model = AsianBartForConditionalGeneration.from_pretrained(
            "hyunwoongko/asian-bart-ko")

        tokens = tokenizer.prepare_seq2seq_batch(
            src_texts="그는 가장 <mask>한 사람이다.",
            src_langs="ko_KR",
        )

        output = model.generate(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            decoder_start_token_id=tokenizer.lang_code_to_id["ko_KR"],
        )

        print(tokenizer.decode(output.tolist()[0]))
        # ko_KR<s> 가장 존경하는 사람 중 한 사람이다.</s>
