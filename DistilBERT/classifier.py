import json
with open("config.json") as json_file:
	config = json.load(json_file)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class BERTClassifier():

	def __init__(self):
		self.tokenizer = AutoTokenizer.from_pretrained(
			"distilbert-base-uncased-finetuned-sst-2-english")
		self.model = AutoModelForSequenceClassification.from_pretrained(
			"distilbert-base-uncased-finetuned-sst-2-english")

	def predict(self, text):
		tokens = self.tokenizer(text, max_length = config["MAX_TOKENS_LEN"], 
			padding = True, return_tensors = "pt")

		with torch.no_grad():
			probabilities = F.softmax(self.model(**tokens)['logits'], dim=1)

		confidence, predicted_class = torch.max(probabilities, dim=1)
		predicted_class = predicted_class.cpu().item()
		probabilities = probabilities.flatten().cpu().numpy().tolist()

		return (
			config["CLASS_NAMES"][predicted_class],
			confidence,
			dict(zip(config["CLASS_NAMES"], probabilities)),
		)

bert = BERTClassifier()

def get_bert():
	return bert