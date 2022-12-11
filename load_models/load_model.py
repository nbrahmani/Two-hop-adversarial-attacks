import textattack
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

# Path to model directory
model_path = ("")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
