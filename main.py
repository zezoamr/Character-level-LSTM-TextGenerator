import time
import torch
import torch.nn as nn
import string
import random
import unidecode
from tensorboardX import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get characters from string.printable
all_characters = string.printable
n_characters = len(all_characters)

# Read large text file (Note can be any text file: not limited to just names)
#filename = "data/names.txt"
filename = 'data/shakespeare_tiny.txt'
file = unidecode.unidecode(open(filename).read())

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, number_layers, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.number_layers = number_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, number_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, input, hidden, cell):
        output = self.embedding(input)
        output, (hidden, cell) = self.lstm(output, (hidden, cell))
        output = output[:, -1, :]
        output = self.fc(output)
        return output, (hidden, cell)
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.number_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.number_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell
    
    def save(self, name):
        torch.save(self.state_dict(), filename[5:] + '-' + name) #removing 'data/' from filename
        
    def load(self, name):
        self.load_state_dict(torch.load(name))
    
class Generator:
    def __init__(self):
        self.chunk_len = 250
        self.num_epochs = 3000
        self.batch_size = 16
        self.print_every = 50
        self.save_every = 1000
        self.hidden_size = 256
        self.num_layers = 2
        self.lr = 0.003
        self.model = LSTM(n_characters, self.hidden_size, self.num_layers, n_characters).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        
    def char_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = all_characters.index(string[c])
        return tensor
    
    def get_random_batch(self):
        text_input = torch.zeros(self.batch_size, self.chunk_len)
        text_target = torch.zeros(self.batch_size, self.chunk_len)

        for i in range(self.batch_size):
            start_idx = random.randint(0, len(file) - self.chunk_len)
            end_idx = start_idx + self.chunk_len + 1
            text_str = file[start_idx:end_idx]
            text_input[i, :] = self.char_tensor(text_str[:-1])
            text_target[i, :] = self.char_tensor(text_str[1:])

        return text_input.long(), text_target.long()
    
    def train(self, load_weights: string = None):
        if load_weights != None:
            self.model.load(load_weights)
            print(f'Model loaded from {load_weights}')
            
        writer = SummaryWriter(f'runs/char-lstm-{int(time.time())}')
        print('Training...')
        
        for epoch in range(self.num_epochs):
            input, target = self.get_random_batch()
            input = input.to(device)
            target = target.to(device)
            hidden, cell = self.model.init_hidden(self.batch_size)
            
            self.optimizer.zero_grad()
            loss = 0
            
            for i in range(self.chunk_len):
                #print("input: ", input, " input shape: ", input.shape, " after:", input[:, i].view(self.batch_size, -1).shape)
                output, (hidden, cell) = self.model(input[:, i].view(self.batch_size, -1), hidden, cell)
                loss += self.criterion(output, target[:, i])
                
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % self.print_every == 0:
                print(f'Epoch: {epoch+1}/{self.num_epochs}, Loss: {loss.item()}')
                print(self.generate(initial_str=random.choice(string.ascii_letters))) #random.choice(string.printable)
                writer.add_scalar('loss', loss.item(), epoch)
                
            output, (hidden, cell) = self.model.forward(input, hidden, cell)
            
            if (epoch + 1) % self.save_every == 0:
                self.model.save(f'char-lstm-{int(time.time())}-epoch{epoch}.pt')
            
    def generate(self, initial_str="A", predict_len=100, temperature=0.8):
        hidden, cell = self.model.init_hidden(1)
        initial_input = self.char_tensor(initial_str)
        predicted = initial_str

        for p in range(len(initial_str) - 1):
            _, (hidden, cell) = self.model(initial_input[p].view(1, 1).to(device), hidden, cell)

        last_char = initial_input[-1]

        for p in range(predict_len):
            output, (hidden, cell) = self.model(last_char.view(1, 1).to(device), hidden, cell)
            output_dist = output.data.view(-1).div(temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
            predicted_char = all_characters[top_char]
            predicted += predicted_char
            last_char = self.char_tensor(predicted_char)

        return predicted
    
genn = Generator()
genn.train()
#genn.train("shakespeare_tiny.txt-char-lstm-1695903193-epoch2999.pt")