import sys
import itertools

import torch
import random

PADDING_IDX = 0
EOS_IDX = 1

flat = itertools.chain.from_iterable

class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, hidden_size, padding_idx=PADDING_IDX, eos_idx=EOS_IDX):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.eos_idx = eos_idx

        self.__build_model()

    def __build_model(self):
        self.word_embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx
         )

        self.lstm = torch.nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.initial_state = (
            torch.autograd.Variable(torch.randn(self.num_layers, self.hidden_size)),
            torch.autograd.Variable(torch.randn(self.num_layers, self.hidden_size))
        )

        self.decoder = torch.nn.Linear(self.hidden_size, self.vocab_size)

    def get_initial_state(self, batch_size):
        init_a, init_b = self.initial_state
        return torch.stack([init_a]*batch_size, dim=-2), torch.stack([init_b]*batch_size, dim=-2)

    def encode(self, X, lengths):
        # X is of shape B x T 
        batch_size, seq_len = X.shape
        assert batch_size == len(lengths)
        embedding = self.word_embedding(X)
        output, hidden = self.lstm(embedding, self.get_initial_state(batch_size))
        return output, hidden 

    def decode(self, h):
        return torch.log_softmax(self.decoder(h), -1)

    def lm_loss(self, Y, Y_hat):
        # Y contains the target token indices. Shape B x T
        # Y_hat contains distributions. Shape B x T x V
        Y_flat = Y.view(-1) # flatten to B*T
        Y_hat_flat = Y_hat.view(-1, self.vocab_size) # flatten to B*T x V
        mask = (Y_flat != self.padding_idx).float()
        num_tokens = torch.sum(mask).item()
        Y_hat_masked = Y_hat_flat[range(Y_hat_flat.shape[0]), Y_flat] * mask
        ce_loss = -torch.sum(Y_hat_masked) / num_tokens
        return ce_loss

def pad_sequences(xs, padding_idx=PADDING_IDX):
    batch_size = len(xs)
    lengths = [len(x) for x in xs]
    longest = max(lengths)
    padded = torch.ones(batch_size, longest).long() * padding_idx
    for i, length in enumerate(lengths):
        sequence = xs[i]
        padded[i, 0:length] = torch.Tensor(sequence[:length])
    return padded, lengths

def train_lm(lstm, data, print_every=10, num_epochs=1000, batch_size=None, **kwds):
    # Inputs should have EOS tokens
    if batch_size is None:
        batch_size = len(data)    
    opt = torch.optim.Adam(params=lstm.parameters(), **kwds)
    for i in range(num_epochs):
        opt.zero_grad()
        batch = random.sample(data, batch_size) # shape B x T
        padded_batch, lengths = pad_sequences(batch)
        z, _ = lstm.encode(padded_batch, lengths) # shape B x T x H
        y_hat = lstm.decode(z) # shape B x T x V
        # x_t -> h_t -> y_t which predicts x_{t+1}
        y = torch.roll(padded_batch, -1, -1) # shape B x T
        loss = lstm.lm_loss(y, y_hat)
        loss.backward()
        opt.step()
        if i % print_every == 0:
            print("epoch %d, loss = %s" % (i, str(loss.item())))
    return lstm

def example(**kwds):
    data = [
        [2,2,EOS_IDX],
        [3,3,3,EOS_IDX],
        [4,4,4,4,EOS_IDX],
        [5,5,5,5,5,EOS_IDX],
        [6,6,6,6,6,6,EOS_IDX],
    ]
    vocab_size = len(data) + 2
    lstm = LSTM(vocab_size, vocab_size, 2, 10)
    return train_lm(lstm, data, **kwds)

class Indexer:
    def __init__(self, eos_idx=EOS_IDX, padding_idx=PADDING_IDX):
        self.counter = itertools.count()
        self.seen = {}
        self.eos_idx = eos_idx
        self.padding_idx = padding_idx
        
    def index_for(self, x):
        if x in self.seen:
            return self.seen[x]
        else:
            result = next(self.counter)
            while result == self.eos_idx or result == self.padding_idx:
                result = next(self.counter)
            self.seen[x] = result
            return result
        
def format_sequences(xss):
    indexer = Indexer()
    for xs in xss:
        yield list(map(indexer.index_for, xs))

def read_unimorph(filename, field=1):
    with open(filename) as infile:
        for line in infile:
            if line.strip():
                parts = line.strip().split("\t")
                yield parts[field].casefold()

def train_unimorph_lm(lang, hidden_size=100, num_layers=2, **kwds):
    data = list(format_sequences(read_unimorph("/Users/canjo/data/unimorph/%s" % lang)))
    vocab_size = len(set(flat(data))) + 2
    lstm = LSTM(vocab_size, vocab_size, num_layers, hidden_size)
    return train_lm(lstm, data, **kwds)

if __name__ == '__main__':
    train_unimorph_lm(*sys.argv[1:])
   
