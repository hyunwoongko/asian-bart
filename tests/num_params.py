from asian_bart import AsianBartForConditionalGeneration


def num_params(model):
    return sum(p.numel() for p in model.parameters())

print(num_params(AsianBartForConditionalGeneration.from_pretrained("hyunwoongko/asian-bart-ecjk")))
print(num_params(AsianBartForConditionalGeneration.from_pretrained("hyunwoongko/asian-bart-en")))
print(num_params(AsianBartForConditionalGeneration.from_pretrained("hyunwoongko/asian-bart-ko")))
print(num_params(AsianBartForConditionalGeneration.from_pretrained("hyunwoongko/asian-bart-ja")))
print(num_params(AsianBartForConditionalGeneration.from_pretrained("hyunwoongko/asian-bart-zh")))
