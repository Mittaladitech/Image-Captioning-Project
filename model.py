import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)      
        self.lstm = nn.LSTM(input_size = embed_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        self.hidden2tag = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        batch_size = features.shape[0]
        embeds = self.word_embeddings(captions[:, :-1])
        lstm_input = torch.cat((features.unsqueeze(dim = 1), embeds), dim=1)
        lstm_out,_ = self.lstm(lstm_input)
        tag_scores = self.hidden2tag(lstm_out)

        return tag_scores

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_ids = []
        for i in range(max_len):                                 
            hidden, states = self.lstm(inputs, states)
            outputs = self.hidden2tag(hidden.squeeze(1))
            pred = outputs.max(1)[1]
            sampled_ids.append(pred.item())
            inputs = self.word_embeddings(pred)
            inputs = inputs.unsqueeze(1)
        return sampled_ids