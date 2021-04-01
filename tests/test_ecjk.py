from unittest import TestCase
from asian_bart import AsianBartTokenizer, AsianBartForConditionalGeneration


class TestECJK(TestCase):

    def test_tokenizer(self):
        tokenizer = AsianBartTokenizer.from_pretrained(
            "hyunwoongko/asian-bart-ecjk")

        tokens = tokenizer.prepare_seq2seq_batch(
            src_texts="안녕하세요.",
            src_langs="ko_KR",
            tgt_texts="hello.",
            tgt_langs="en_XX",
        )

        print(tokens)

    def test_model(self):
        tokenizer = AsianBartTokenizer.from_pretrained(
            "hyunwoongko/asian-bart-ecjk")

        model = AsianBartForConditionalGeneration.from_pretrained(
            "hyunwoongko/asian-bart-ecjk")

        tokens = tokenizer.prepare_seq2seq_batch(
            src_texts="Kevin is the <mask> man in the world.",
            src_langs="en_XX",
        )

        output = model.generate(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"],
        )

        print(tokenizer.decode(output.tolist()[0]))
        # en_XX<s> Kevin is the most beautiful man in the world.</s>
