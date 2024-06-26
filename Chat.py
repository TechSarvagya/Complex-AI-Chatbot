from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
#import nltk
from englisttohindi.englisttohindi import EngtoHindi
import inltk
from inltk.inltk import setup
from inltk.inltk import tokenize
import random
from simplemma import lemmatize
setup("hi")

with open("Intents.json", 'r',encoding="utf8") as f:
    intents = json.load(f)
'''
lt = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')
'''

#def w_tokenize(sentence):
#   return tokenize(sentence)

def lemm(word):
    return lemmatize(word.lower(),lang="hi")


def bag_of_words(tks, all_words):
    tks = [lemm(w) for w in tks if w not in ignore_words]
    bag = np.zeros(len(all_words))
    for i, word in enumerate(all_words):
        if word in tks:
            bag[i] = 1.0
    return bag
all_words = []
tags = []
xy = []
ignore_words = ['!', '?', '.', ',']

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        tokenized_sent = tokenize(pattern,language_code="hi")
        lem_sent = [lemm(w) for w in tokenized_sent if w not in ignore_words]
        all_words.extend(lem_sent)
        xy.append((lem_sent, tag))

all_words = sorted(set(all_words))
print(all_words)
tags = sorted(set(tags))
X_train = []
y_train = []
for (pattern, tag) in xy:
    bag = bag_of_words(pattern, all_words)
    X_train.append(bag)
    y_train.append(tags.index(tag))
X_train = np.array(X_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.n_samples


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out


data = ChatDataset()
train_loader = DataLoader(dataset=data, batch_size=8, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(len(X_train[0]), 8, len(tags)).to(device)
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), 0.001)
for epoch in range(100):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        outputs = model(words.to(torch.float32))
        loss = criterion(outputs, labels.type(torch.LongTensor))
        opt.zero_grad()
        loss.backward()
        opt.step()
    if (epoch + 1) % 100 == 0:
        print(f"epoch {epoch + 1}/100,loss={loss.item():.4f}")
print(f"final loss={loss.item():.4f}")

d = {'model_state': model.state_dict(),
     'input_size': X_train[0],
     'hidden_size': 8,
     'output_size': len(tags),
     'all_words': all_words,
     'tags': tags}
torch.save(d, "data.pth")
model.load_state_dict(d['model_state'])
model.eval()
# print("Type Hi/Hello to start the chat and quit to exit")
def get_response(sentence):
    sentence=EngtoHindi(sentence).convert
    print(f"Translated:{sentence}")
    sentence = tokenize(sentence,language_code="hi")
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    output = model(X.to(torch.float32))
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob >= 0.50:
        for intent in intents['intents']:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                return (tag,response)

    else:

        return ("None",EngtoHindi("I am sorry I could not understand you").convert)