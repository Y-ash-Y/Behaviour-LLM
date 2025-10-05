from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

TOKEN = 'bhadresh-savani/distilbert-base-uncased-emotion'  # small HF emotion model

tokenizer = AutoTokenizer.from_pretrained(TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(TOKEN)

def predict(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    labels = [model.config.id2label[i] for i in range(len(probs))]
    return dict(zip(labels, probs))

if __name__ == '__main__':
    print(predict('I feel frustrated and tired.'))
