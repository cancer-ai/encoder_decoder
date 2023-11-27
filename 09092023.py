import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_metric_learning import losses

import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz

class CAI_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(CAI_LSTM, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

    def forward(self, x):
        encoded_output, (hidden, cell) = self.encoder(x)
        decoded_output, _ = self.decoder(encoded_output, (hidden, cell))
        return decoded_output

    def decode(self, x): 
            pass
        
input_size = 512
hidden_size = 512
num_layers = 2
learning_rate = 0.01

model = CAI_LSTM(input_size, hidden_size, num_layers)
filename =  "/project/mayocancerai/GenSLM/07_18_2023_wolff_embedding_output.txt"

with open(filename) as f:
    lines = f.readlines()
    print(len(lines))
    all_embeddings = []
    for line in lines:
        embedding = [] 
        split = line.split(" ")
        for number in split: 
            embedding.append(float(number))
        all_embeddings.append(embedding)
        
model_path = "08292023_model"

model.load_state_dict(torch.load(model_path))
integrated_gradients = IntegratedGradients(model)
attributions_ig = integrated_gradients.attribute(torch.Tensor(all_embeddings))
print(attributions_ig)
