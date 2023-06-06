import torch
import torch.nn as nn
import torch.optim as optim

from utils import *
from embedding import embed

# LSTM model definition
'''class CAI_LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_layer_size=512, num_layers=1, dropout=0.5, batch_size=1, output_size=1):
        super(CAI_LSTM self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout)
        self.tanh = nn.Tanh()

    def forward(self, input_seq):
        lstm_out, self.hidden = self.lstm(input_seq.view(-1,len(input_seq[0]),len(input_seq[0][0])), self.hidden)
        pred = lstm_out.view(-1,len(lstm_out[0]),len(lstm_out[0][0][:,-1]))
        pred = self.tanh(pred)
        return pred
 '''       

class CAI_LSTM(nn.Module):
    '''def __init__(self, input_size=2, hidden_layer_size=512, num_layers=1, dropout=0.5, batch_size=1, output_size=1):
        super().__init__()

        # architecture
        self.enc = encoder(input_size, hidden_layer_size)
        self.dec = decoder(output_size)
        self = self.cuda() if CUDA else self

    def forward(self, input_seq): # for training
    
        c = self.conv(input1)
        f = self.fc1(input2)
        
        encoder = self.encoder(input_seq)
        decoder = self.decoder(decode(input_seq))
        '''
        # now we can reshape `c` and `f` to 2D and concat them
        #combined = torch.cat((c.view(c.size(0), -1),
                              f.view(f.size(0), -1)), dim=1)
        #out = self.fc2(combined)
    
        #b = y0.size(0) # batch size
        #loss = 0
        #self.zero_grad()
        #mask, lens = maskset(xw)
        #self.dec.M = self.enc(b, xc, xw, lens)
        #self.dec.hidden = self.enc.hidden
        #self.dec.attn.Va = zeros(b, 1, HIDDEN_SIZE)
        #yi = LongTensor([SOS_IDX] * b)
        #for t in range(y0.size(1)):
        #    yo = self.dec(yi.unsqueeze(1), mask)
        #    yi = y0[:, t] # teacher forcing
        #    loss += F.nll_loss(yo, yi, ignore_index = PAD_IDX)
        #loss /= y0.size(1) # divide by senquence length
        #return loss

        def __init__(self, input_size, hidden_size, num_layers):
            super(CAI_LSTM, self).__init__()
            self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

        def forward(self, x):
            # Encoder
            encoded_output, (hidden, cell) = self.encoder(x)
            
            # Decoder
            decoded_output, _ = self.decoder(encoded_output, (hidden, cell))
            
            return decoded_output
        

        def decode(self, x): # for inference
            pass
# Define hyperparameters
input_size = 10
hidden_size = 5
num_layers = 2
learning_rate = 0.01
num_epochs = 100

# Create model
model = CAI_LSTM(input_size, hidden_size, num_layers)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Define EMD loss function
def emd_loss(x, y):
    distance_matrix = torch.cdist(x, y, p=1)
    row_indices, col_indices = torch.linear_sum_assignment(distance_matrix)
    loss = torch.mean(distance_matrix[row_indices, col_indices])
    return loss

# Create dummy data
input_data = torch.randn(32, 10, 10) # Batch size x sequence length x input size

# Train model
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(input_data)
    
    # Calculate loss
    input_data_permuted = input_data[torch.randperm(input_data.size()[0])]
    outputs_permuted = outputs[torch.randperm(outputs.size()[0])]
    loss = emd_loss(outputs, input_data_permuted)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress
    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

'''class encoder(nn.Module):
    def __init__(self, cti_size, wti_size):
        super().__init__()
        self.hidden = None # encoder hidden states

        # architecture
        self.embed = embed(ENC_EMBED, cti_size, wti_size)
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = self.embed.dim,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )

    def init_state(self, b): # initialize RNN states
        n = NUM_LAYERS * NUM_DIRS
        h = HIDDEN_SIZE // NUM_DIRS
        hs = zeros(n, b, h) # hidden state
        if RNN_TYPE == "LSTM":
            cs = zeros(n, b, h) # LSTM cell state
            return (hs, cs)
        return hs

    def forward(self, b, xc, xw, lens):
        self.hidden = self.init_state(b)
        x = self.embed(xc, xw)
        x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first = True)
        h, _ = self.rnn(x, self.hidden)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)
        return h

class decoder(nn.Module):
    def __init__(self, wti_size):
        super().__init__()
        self.M = None # source hidden states
        self.hidden = None # decoder hidden states

        # architecture
        self.embed = embed(DEC_EMBED, 0, wti_size)
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = self.embed.dim + HIDDEN_SIZE, # input feeding
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )
        //self.attn = attn()
        self.out = nn.Linear(HIDDEN_SIZE, wti_size)
        self.tanh = nn.Tanh()

    def forward(self, y1, mask):
        x = self.embed(None, y1)
        x = torch.cat((x, self.attn.Va), 2) # input feeding
        h, _ = self.rnn(x, self.hidden)
        #h = self.attn(h, self.M, mask)
        h = self.out(h).squeeze(1)
        y = self.softmax(h)
        return y

'''
