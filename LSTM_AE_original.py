import torch
import torch.nn as nn
import torch.optim as optim

class AutoencoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(AutoencoderLSTM, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

    def forward(self, x):
        # Encoder
        encoded_output, (hidden, cell) = self.encoder(x)
        
        # Decoder
        decoded_output, _ = self.decoder(encoded_output, (hidden, cell))
        
        return decoded_output

# Define hyperparameters
input_size = 10
hidden_size = input_size
num_layers = 4
learning_rate = 0.01
num_epochs = 300000
batch_size = 5120
sequence_length = 2

# Create model
model = AutoencoderLSTM(input_size, hidden_size, num_layers)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define MMD loss function
def mmd_loss(x, y, kernel_size=10):
    x_kernel = torch.exp(-torch.sum((x.unsqueeze(1) - x)**2, dim=2) / (2 * kernel_size ** 2))
    y_kernel = torch.exp(-torch.sum((y.unsqueeze(1) - y)**2, dim=2) / (2 * kernel_size ** 2))
    xy_kernel = torch.exp(-torch.sum((x.unsqueeze(1) - y)**2, dim=2) / (2 * kernel_size ** 2))
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

criterion = nn.MSELoss()


# Train model
for epoch in range(num_epochs):
    # Create dummy data
    input_data = torch.rand(batch_size, sequence_length, input_size) # Batch size x sequence length x input size

    # Forward pass
    outputs = model(input_data)
    
    # Calculate MMD loss
    #input_data_permuted = input_data[torch.randperm(input_data.size()[0])]
    #outputs_permuted = outputs[torch.randperm(outputs.size()[0])]
    #loss = mmd_loss(outputs, input_data_permuted)

    loss = criterion(outputs,input_data)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress
    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        print("Input[0]:",input_data[0])
        print("Outputs[0]:",outputs[0])
