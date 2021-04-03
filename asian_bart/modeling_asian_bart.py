from transformers import (
    MBartModel,
    MBartForConditionalGeneration,
    MBartForQuestionAnswering,
    MBartForSequenceClassification,
    MBartForCausalLM,
    MBartConfig,
)


class AsianBartModel(MBartModel):
    pass


class AsianBartForConditionalGeneration(MBartForConditionalGeneration):
    pass


class AsianBartForQuestionAnswering(MBartForQuestionAnswering):
    pass


class AsianBartForSequenceClassification(MBartForSequenceClassification):
    pass


class AsianBartForCausalLM(MBartForCausalLM):
    pass


class AsianBartConfig(MBartConfig):
    pass
