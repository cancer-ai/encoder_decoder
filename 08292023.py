import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_metric_learning import losses

import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, inputs, targets):
        #loss = -1 * (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        #return loss.mean()
        diff = torch.sub(inputs, targets, alpha=1)
        abs_diff = diff.abs()
        loss = abs_diff.sum()/512
        return loss 

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
learning_rate = 0.01

#model = CAI_LSTM(input_size, hidden_size, num_layers)
filename =  "/project/mayocancerai/GenSLM/07_18_2023_wolff_embedding_output.txt"

with open(filename) as f:
    lines = f.readlines()
    print(len(lines))
    all_embeddings = []
    for line in lines:
        embedding = [] 
        split = line.split(" ")
        #print(split)
        for number in split: 
            embedding.append(float(number))
        #print(embedding)
        all_embeddings.append(embedding)
    #print(all_embeddings)

#loss_function = nn.CrossEntropyLoss()

print("PyTorch Version: " + str(torch.__version__))
print("Cuda Available: " + str(torch.cuda.is_available()))
#print("Device Name: " + str(torch.cuda.get_device_name(0)))
#print("Device Count: " + str(torch.cuda.device_count()))
#print("Device(0): " + str(torch.cuda.device(0)))
#print("Device Current: " + str(torch.cuda.current_device()))'''

def train_gpu_model(model, num_layers, loss_function, optimizer, epochs): 
    if torch.cuda.is_available():
    
        print("gpu")
        #my_device = torch.device('cuda')
        x = torch.Tensor(all_embeddings)
        x = x.to('cuda')
        y = x
        y = y.to('cuda')
        model = model.to('cuda')
        
        best_loss = 100000
        #model_path = "09282023_model_layers_" + str(num_layers) + "_epochs_" + str(epochs)
        
        print("Epoch, Loss")
        for i in range(epochs): 
            model.train(True)
            #model.eval()
           
            output = model(x)
            loss = loss_function(output, y)
            if i % 100 == 1:
                print("{}, {}".format(i,loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < best_loss: 
                best_loss = loss
                #torch.save(model.state_dict(), model_path)
        print("num layers: "+ str(num_layers) + " epochs: " + str(epochs) + " best loss: " + str(best_loss))
        '''
        
    
        model.load_state_dict(torch.load(model_path))
        integrated_gradients = IntegratedGradients(model)
        inputs = all_embeddings[0]
        inputs = torch.Tensor([inputs])
        inputs = inputs.to("cuda")
        #inputs = inputs[[0]][[0]]
        print(inputs)
        #for i in range(len(inputs)):
        #    inputs = torch.Tensor(inputs[i])
        attributions_ig = integrated_gradients.attribute(inputs)
        print(attributions_ig)'''
        '''else:
            print("no gpu")
        '''
for i in range(2, 10, 2): 
    model = CAI_LSTM(input_size, hidden_size, num_layers = i)
    loss_function = CustomLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)  
    for e in [1000, 2000, 3000, 5000, 10000]: 
        train_gpu_model(model, i, loss_function, optimize, epochs = e)
         
