import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize(
    text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    import re
    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]
    return [t.split()[:max_length] for t in text]

def build_index_map(
    word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    if max_words:
        sorted_counts = sorted_counts[:max_words-1]
    sorted_words = ["[PAD]"] + [item[0] for item in sorted_counts]
    return {word: ix for ix, word in enumerate(sorted_words)}

# modify build_word_counts for SNLI
# so that it takes into account batch['premise'] and batch['hypothesis']
def build_word_counts(dataloader) -> "dict[str, int]":
    word_counts = {}
    for batch in dataloader:
        for words in batch:
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts

def tokens_to_ix(
    tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]

class CharSeqDataloader():
    def __init__(self, filepath, seq_len, examples_per_epoch):
        with open(filepath, 'r') as file:
            data = file.read()
        self.data = list(data) #a list of the file as chars (inlcudes spaces and \n and wtv else)
        
        self.unique_chars = list(set(self.data))
        self.vocab_size = len(self.unique_chars)
        self.mappings = self.generate_char_mappings(self.unique_chars) # fill in
        self.seq_len = seq_len # The sequence length (set as an instance variable to be accessed later)
        self.examples_per_epoch = examples_per_epoch 
    
    def generate_char_mappings(self, uq):
        # uq = The list of unique characters to create mappings for.

        # A dict mapping characters to their corresponding indices
        char_to_idx = {char: idx for idx, char in enumerate(uq)}
        
        # A dict mapping indices to their corresponding characters
        idx_to_char = {idx: char for idx, char in enumerate(uq)}
        
        dicty = {'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}
        return dicty

    def convert_seq_to_indices(self, seq):
        # seq = List[str]
        # return = List[int] --> need char_to_idx key!
        return [self.mappings['char_to_idx'][char] for char in seq]

    def convert_indices_to_seq(self, seq):
        # seq = List[int]
        # return = List[str] --> need idx_to_char key!
        return [self.mappings['idx_to_char'][idx] for idx in seq]


    def get_example(self):
        data_to_indices = self.convert_seq_to_indices(self.data)
        
        for _ in range(self.examples_per_epoch):
            # input sequence will be a random **consecutive** sequence of characters (size seq_len) taken from dataset
            # max of random idx cannot be higher than len(data) - seq_len to allow for us to extract input seq starting from that idx
            beg_index = random.randint(0, len(data_to_indices) - self.seq_len - 1)
            # input from beg_index to how much we need to take after that
            input_seq = data_to_indices[beg_index : beg_index + self.seq_len]
            # target_seq for the RNN to predict will be the input_seq shifted over by 1.
            target_seq = data_to_indices[beg_index + 1:beg_index + self.seq_len + 1]
            # to tensor
            yield torch.tensor(input_seq), torch.tensor(target_seq)



class CharRNN(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_chars = n_chars
        self.embedding_size = embedding_size
        self.embedding_layer = nn.Embedding(num_embeddings=self.n_chars, embedding_dim=self.embedding_size)
        # input-hidden
        self.wax = nn.Linear(embedding_size, hidden_size, bias=True) 
        # hidden-hidden
        self.waa = nn.Linear(hidden_size, hidden_size, bias=False)    
        # hidden-output
        self.wya = nn.Linear(hidden_size, n_chars, bias=True)        
        
    def rnn_cell(self, i, h):
        i = i.to(device)
        h = h.to(device)
        combo = self.wax(i) + self.waa(h)
        a_t = torch.tanh(combo) 
        y_t = self.wya(a_t)
        return y_t, a_t

    def forward(self, input_seq, hidden = None):
        input_seq = input_seq.to(device)
        if hidden is None:
            hidden = torch.zeros(self.hidden_size)  
        embedded_seq = self.embedding_layer(input_seq)
        outputs = []
        
        # run for length of input sequence
        for i in range(embedded_seq.shape[0]):
            # get current inout char from the seq in emb form
            current_input = embedded_seq[i]
            # run the rnn cell and get the iutput and current hidden state to pass next iter
            y_t, hidden = self.rnn_cell(current_input, hidden)
            # save the cell output in list
            outputs.append(y_t)
            hidden = hidden.detach()

        out = torch.stack(outputs)
        return out, hidden

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)
    
    def sample_sequence(self, starting_char, seq_len, temp=0.5, top_k=None, top_p=None):
        #including starting char in the output sequence
        generated_seq = [starting_char]
        hidden = None
        #using forward, need to pass it as tensor sequence of chars not single char
        input_seq = torch.tensor([starting_char])
        
        
        for _ in range(seq_len):
            output, hidden = self.forward(input_seq, hidden)
            # have to squeeze the output here
            out_distribution = F.softmax(output.squeeze() / temp, dim=-1)
            out_distribution = out_distribution.unsqueeze(0)
            if top_k is not None:
                out_distribution = top_k_filtering(out_distribution, top_k=top_k)
            elif top_p is not None:
                out_distribution = top_p_filtering(out_distribution, top_p=top_p)
            
            #sample from categorical and add to the output sequence
            sample = torch.distributions.Categorical(logits=out_distribution).sample().item()
            generated_seq.append(sample)
            input_seq = torch.tensor([[sample]])
            
        
        return generated_seq

class CharLSTM(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_chars = n_chars

        self.embedding_layer = nn.Embedding(n_chars, embedding_size)
        
        self.forget_gate = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(embedding_size + hidden_size, hidden_size)
    
        self.cell_state_layer = nn.Linear(embedding_size + hidden_size, hidden_size)
        # output
        self.fc_output = nn.Linear(hidden_size, n_chars)

    def forward(self, input_seq, hidden = None, cell = None):
        if hidden is None:
            hidden = torch.zeros(self.hidden_size)
        if cell is None:
            cell = torch.zeros(self.hidden_size)
        embedded_seq = self.embedding_layer(input_seq)
        outputs = []
        for i in range(embedded_seq.shape[0]):
            current_input = embedded_seq[i]
            o, hidden, cell = self.lstm_cell(current_input, hidden, cell)
            outputs.append(o)
            hidden = hidden.detach()
        
        out = torch.stack(outputs)
        return out, hidden, cell

    def lstm_cell(self, i, h, c):
        i = i.to(device)
        h = h.to(device)
        c = c.to(device)
        combo_in_hid = torch.cat((i, h), dim=-1)
    
        #forget gate
        forget_gatey = torch.sigmoid(self.forget_gate(combo_in_hid))
        #input gate
        input_gatey = torch.sigmoid(self.input_gate(combo_in_hid))
        #output gate
        output_gatey = torch.sigmoid(self.output_gate(combo_in_hid))
        #cell state layer
        cell_state = torch.tanh(self.cell_state_layer(combo_in_hid))
        
        # c=prev cell state
        c_new = forget_gatey * c + input_gatey * cell_state
        h_new = output_gatey * torch.tanh(c_new)
        o = self.fc_output(h_new)

        return o, h_new, c_new
    
    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)
    
    def sample_sequence(self, starting_char, seq_len, temp=0.5, top_k=None, top_p=None):
        generated_seq = [starting_char]
        hidden = None
        cell = None
        #using forward, need to pass it as tensor sequence of chars not single char
        input_seq = torch.tensor([starting_char])
        
        for _ in range(seq_len):
            output, hidden, cell = self.forward(input_seq, hidden, cell)
            # have to squeeze the output here
            out_distribution = F.softmax(output.squeeze() / temp, dim=-1)
            out_distribution = out_distribution.unsqueeze(0)
            if top_k is not None:
                out_distribution = top_k_filtering(out_distribution, top_k=top_k)
            elif top_p is not None:
                out_distribution = top_p_filtering(out_distribution, top_p=top_p)
            
            #sample from categorical and add to the output sequence
            sample = torch.distributions.Categorical(out_distribution).sample().item()
            generated_seq.append(sample)
            input_seq = torch.tensor([sample])
        
        return generated_seq
    


def top_k_filtering(logits, top_k=40):
    _, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    top_k_indices = sorted_indices[:, :top_k]
    # create the tensor mask
    mask = torch.zeros_like(logits, dtype=torch.bool)
    # fill with true for indices of topk
    mask.scatter_(-1, top_k_indices, True)
    # fill outside of topk with -inf
    filetered_logits = logits.masked_fill(~mask, float('-inf'))
#     print(filetered_logits)
    return filetered_logits

def top_p_filtering(logits, top_p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    # get the probs for all these sorted logits--> softmax and then cumsum
    cum_prob = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    exceed_indices  = cum_prob > top_p
    exceed_indicess = exceed_indices.clone()
    exceed_indices[:, 1:] = exceed_indicess[:, :-1]
    exceed_indices[:, 0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(1, sorted_indices, exceed_indices)
    logits[mask] = float('-inf')
    return logits


def train(model, dataset, lr, out_seq_len, num_epochs):

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to GPU
    
    torch.autograd.set_detect_anomaly(True)
    optimizer = model.get_optimizer(lr)
    loss_fn = model.get_loss_function()

    for epoch in range(num_epochs):
        hidden = None
        running_loss = 0
        n = 0
        for in_seq, out_seq in dataset.get_example():
            # main loop code
            # in_seq = torch.tensor(in_seq).unsqueeze(0).to(device)
            # out_seq = torch.tensor(out_seq).unsqueeze(0).to(device)
            in_seq = in_seq.to(device)
            out_seq = out_seq.to(device)
            out, hidden = model(in_seq, hidden)
        
            optimizer.zero_grad()
            loss = loss_fn(out, out_seq)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n += 1

        # print info every X examples
        print(f"Epoch {epoch}. Running loss so far: {(running_loss/n):.8f}")

        print("\n-------------SAMPLE FROM MODEL-------------")

        # code to sample a sequence from the model randomly

        with torch.no_grad():
            # just sampling a random index from all the chars we have
            beg_idx = random.randint(0, model.n_chars - 1)
            # call sample_sequence and grab the generated seq
            generated_seq = model.sample_sequence(beg_idx, out_seq_len)
            output_text = ''.join([dataset.mappings['idx_to_char'][idx] for idx in generated_seq])
            print("Model spat out:", output_text)

        print("\n------------/SAMPLE FROM MODEL/------------")
    
    return model # return model optionally



def run_char_rnn():
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 0.002
    num_epochs = 100
    epoch_size = 130 # one epoch is this # of examples
    out_seq_len = 200
    data_path = "../input/a3-data/data/shakespeare.txt"

    # code to initialize dataloader, model
    dataset = CharSeqDataloader(data_path, seq_len, epoch_size)
    model = CharRNN(dataset.vocab_size, embedding_size, hidden_size).to(device)

    # Train the model
    trained_model = train(model, dataset, lr=lr, 
                out_seq_len=out_seq_len, 
                num_epochs=num_epochs)
    
    # Generate samples from the trained model
    print("### Samples with temp=0.5")
    for _ in range(3):  # Generate 3 samples
        starting_char = random.choice(dataset.unique_chars)  # Start with a random character
        generated_seq = trained_model.sample_sequence(dataset.mappings['char_to_idx'][starting_char], out_seq_len, temp=0.5, top_k=None, top_p=None)
        output_text = ''.join([dataset.mappings['idx_to_char'][idx] for idx in generated_seq])
        print("\nGenerated Sequence:")
        print(output_text)
    
    print("### Samples with temp=0.5, top_p = 0.3")
    for _ in range(3):  # Generate 3 samples
        starting_char = random.choice(dataset.unique_chars)  # Start with a random character
        generated_seq = trained_model.sample_sequence(dataset.mappings['char_to_idx'][starting_char], out_seq_len, temp=0.5, top_k=None, top_p=0.3)
        output_text = ''.join([dataset.mappings['idx_to_char'][idx] for idx in generated_seq])
        print("\nGenerated Sequence:")
        print(output_text)
        
    print("### Samples with temp=0.5, top_p = 0.9")
    for _ in range(3):  # Generate 3 samples
        starting_char = random.choice(dataset.unique_chars)  # Start with a random character
        generated_seq = trained_model.sample_sequence(dataset.mappings['char_to_idx'][starting_char], out_seq_len, temp=0.5, top_k=None, top_p=0.9)
        output_text = ''.join([dataset.mappings['idx_to_char'][idx] for idx in generated_seq])
        print("\nGenerated Sequence:")
        print(output_text)
        
    print("### Samples with temp=0.5, top_k = 40")
    for _ in range(3):  # Generate 3 samples
        starting_char = random.choice(dataset.unique_chars)  # Start with a random character
        generated_seq = trained_model.sample_sequence(dataset.mappings['char_to_idx'][starting_char], out_seq_len, temp=0.5, top_k=40, top_p=None)
        output_text = ''.join([dataset.mappings['idx_to_char'][idx] for idx in generated_seq])
        print("\nGenerated Sequence:")
        print(output_text)
        
    print("### Samples with temp=0.5, top_k = 100")
    for _ in range(3):  # Generate 3 samples
        starting_char = random.choice(dataset.unique_chars)  # Start with a random character
        generated_seq = trained_model.sample_sequence(dataset.mappings['char_to_idx'][starting_char], out_seq_len, temp=0.5, top_k=100, top_p=None)
        output_text = ''.join([dataset.mappings['idx_to_char'][idx] for idx in generated_seq])
        print("\nGenerated Sequence:")
        print(output_text)
        
    print("### Samples with temp=0.1")
    for _ in range(3):  # Generate 3 samples
        starting_char = random.choice(dataset.unique_chars)  # Start with a random character
        generated_seq = trained_model.sample_sequence(dataset.mappings['char_to_idx'][starting_char], out_seq_len, temp=0.1, top_k=None, top_p=None)
        output_text = ''.join([dataset.mappings['idx_to_char'][idx] for idx in generated_seq])
        print("\nGenerated Sequence:")
        print(output_text)
        
    print("### Samples with temp=0.9")
    for _ in range(3):  # Generate 3 samples
        starting_char = random.choice(dataset.unique_chars)  # Start with a random character
        generated_seq = trained_model.sample_sequence(dataset.mappings['char_to_idx'][starting_char], out_seq_len, temp=0.9, top_k=None, top_p=None)
        output_text = ''.join([dataset.mappings['idx_to_char'][idx] for idx in generated_seq])
        print("\nGenerated Sequence:")
        print(output_text)

def run_char_lstm():
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 0.002
    num_epochs = 100
    epoch_size = 10
    out_seq_len = 200
    data_path = "./data/shakespeare.txt"

    # code to initialize dataloader, model
    
    train(model, dataset, lr=lr, 
                out_seq_len=out_seq_len, 
                num_epochs=num_epochs)


def fix_padding(batch_premises, batch_hypotheses):
    
    batch_premises = [torch.tensor(batch) for batch in batch_premises]
    batch_hypotheses = [torch.tensor(batch) for batch in batch_hypotheses]
    
    pad_batch_premises = torch.nn.utils.rnn.pad_sequence(batch_premises, batch_first=True)
    pad_batch_hypotheses = torch.nn.utils.rnn.pad_sequence(batch_hypotheses, batch_first=True)

    #REVERSE
    batch_premises_reversed = [torch.flip(batch, dims=[-1]) for batch in batch_premises]
    pad_batch_premises_reversed = torch.nn.utils.rnn.pad_sequence(batch_premises_reversed, batch_first=True)
    
    batch_hypotheses_reversed = [torch.flip(batch, dims=[-1]) for batch in batch_hypotheses]
    pad_batch_hypotheses_reversed = torch.nn.utils.rnn.pad_sequence(batch_hypotheses_reversed, batch_first=True)

    return pad_batch_premises, pad_batch_hypotheses, pad_batch_premises_reversed, pad_batch_hypotheses_reversed


def create_embedding_matrix(word_index, emb_dict, emb_dim):
    snli_embs = torch.zeros(len(word_index), emb_dim)

    for word, i in word_index.items():
        # is the word present in the glove embeddings?
        if word in emb_dict:
            #copy it over
            word_tensor = torch.tensor(emb_dict[word])
            snli_embs[i] = word_tensor
    return snli_embs

def evaluate(model, dataloader, index_map):
    model.eval()
    correct_preds = 0
    all_data = 0
    # examples_per_epoch = dataloader.examples_per_epoch
    
    for data in dataloader:
        #{'premise': ['A senior is waiting at the window of a restaurant that serves sandwiches.'], 'hypothesis': ['A man is looking to order a grilled cheese sandwich.'], 'label': tensor([1])}
        premise = data['premise']
        hypothesis = data['hypothesis']
        labels = data['label']

        #to indices using index_map -> Dict[str, int] 
        premise_tokens = tokenize(premise)
        print(premise_tokens)
        premise_indices = tokens_to_ix(premise_tokens, index_map)
        hypothesis_tokens = tokenize(hypothesis) 
        hypothesis_indices = tokens_to_ix(hypothesis_tokens, index_map)
        # Pass the indices to the model
        with torch.no_grad():
            output = model(premise_indices, hypothesis_indices)

        pred_values, pred_indices = torch.max(output, 1)

        all_data += labels.size(0)
        correct_preds += (pred_indices == labels).sum().item()

    accuracy = correct_preds / all_data

    return accuracy

class UniLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes):
        super(UniLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        #nn.Embedding layer (called “embedding_layer”). Make sure to set padding_idx
        #correctly. For simplicity, set the embedding dimension to be the
        #same as the hidden dimension of the LSTM
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim, padding_idx=0)
        #pass “batch_first” to LSTM, as we are putting our batches as the first dimension
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.int_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, a, b):

        padded_a, padded_b, _, _ = fix_padding(a, b)

        emb_a = self.embedding_layer(padded_a)
        emb_b = self.embedding_layer(padded_b)

        output_a, (h_n,c_na) = self.lstm(emb_a)
        output_b, (h_n, c_nb) = self.lstm(emb_b)

        # FINAL CELL STATE
        cell_state_a = c_na[-1]
        cell_state_b = c_nb[-1]
        # print("final cell: ", c_na.shape, c_nb.shape )
        # Concat them
        concatenated_states = torch.cat((cell_state_a, cell_state_b), dim=1)
        intermediate_output = F.relu(self.int_layer(concatenated_states))
        output = self.out_layer(intermediate_output)

        return output


class ShallowBiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes):
        super(ShallowBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding_layer = nn.Embedding(vocab_size, hidden_dim)

        self.lstm_forward = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_backward = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.int_layer = nn.Linear(4 * hidden_dim, hidden_dim)
        
        self.out_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, a, b):
        padded_a, padded_b, padded_a_rev, padded_b_rev = fix_padding(a, b)

        emb_a = self.embedding_layer(padded_a)
        emb_b = self.embedding_layer(padded_b)

        emb_a_rev = self.embedding_layer(padded_a_rev)
        emb_b_rev = self.embedding_layer(padded_b_rev)

        #forward
        output_a, (h_n,c_na) = self.lstm_forward(emb_a)
        output_b, (h_n, c_nb) = self.lstm_forward(emb_b)
        #backward
        output_a_rev, (_,c_na_rev) = self.lstm_backward(emb_a_rev)
        output_b_rev, (_,c_nb_rev) = self.lstm_backward(emb_b_rev)

        # final cell states
        cell_state_a = c_na[-1]
        cell_state_b = c_nb[-1]
        cell_state_a_rev= c_na_rev[-1]
        cell_state_b_rev= c_nb_rev[-1]
        
        concatenated_states = torch.cat((cell_state_a, cell_state_a_rev, cell_state_b, cell_state_b_rev), dim=-1)
        intermediate_output = F.relu(self.int_layer(concatenated_states))
        
        output = self.out_layer(intermediate_output)
        
        return output

def run_snli(model):
    dataset = load_dataset("snli")
    glove = pd.read_csv('./data/glove.6B.100d.txt', sep=" ", quoting=3, header=None, index_col=0)

    glove_embeddings = "" # fill in your code

    train_filtered = dataset['train'].filter(lambda ex: ex['label'] != -1)
    valid_filtered = dataset['validation'].filter(lambda ex: ex['label'] != -1)
    test_filtered =  dataset['test'].filter(lambda ex: ex['label'] != -1)

    # code to make dataloaders

    word_counts = build_word_counts(dataloader_train)
    index_map = build_index_map(word_counts)

def run_snli_lstm():
    model_class = "" # fill in the classs name of the model (to initialize within run_snli)
    run_snli(model_class)

def run_snli_bilstm():
    model_class = "" # fill in the classs name of the model (to initialize within run_snli)
    run_snli(model_class)

if __name__ == '__main__':
    run_char_rnn()
    # run_char_lstm()
    # run_snli_lstm()
    # run_snli_bilstm()
