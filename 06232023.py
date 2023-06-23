import torch
import torch.nn as nn
import torch.optim as optim

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
filename =  "/project/mayocancerai/GenSLM/Embedding_output.txt"

with open(filename) as f:
    lines = f.readlines()
    
    all_embeddings = []
    for line in lines:
        embedding = [] 
        split = line.split(" ")
        for number in split: 
            embedding.append(float(number))
        all_embeddings.append(embedding)
        
loss_function = nn.CrossEntropyLoss()   
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)  

x = torch.Tensor(all_embeddings)
y = x

epochs=50000

for i in range(epochs): 
    output = model(x)
    loss = loss_function(output, y)
    if i % 100 == 1:
        print("Epoch number: {} and the loss : {}".format(i,loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
