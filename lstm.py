import sys
import itertools

import numpy as np
import torch
import random

INF = float('inf')

PADDING = '<!PAD!>'
EOS = '<!EOS!>'

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
        return torch.stack([init_a]*batch_size, dim=-2), torch.stack([init_b]*batch_size, dim=-2) # L x B x H

    def encode(self, X):
        # X is of shape B x T 
        batch_size, seq_len = X.shape
        embedding = self.word_embedding(X)
        init_h, init_c = self.get_initial_state(batch_size) # init_h is L x B x H
        output, (out_h, out_c) = self.lstm(embedding, (init_h, init_c)) # output is B x T x H
        # Also need to include the initial state itself
        # And the last state is after consuming EOS, so it doesn't matter
        # So add the initial state in first, and remove the final state
        full_output = torch.cat((init_h[-1].unsqueeze(-2), output), dim=-2)[:, 0:-1, :]
        return full_output

    def decode(self, h):
        logits = self.decoder(h)
        # Remove probability mass from the pad token
        logits[:, :, self.padding_idx] = -INF
        return torch.log_softmax(logits, -1)

    def lm_loss(self, Y, Y_hat):
        # Y contains the target token indices. Shape B x T
        # Y_hat contains distributions. Shape B x (T+1) x V
        Y_flat = Y.view(-1) # flatten to B*T
        Y_hat_flat = Y_hat.view(-1, self.vocab_size) # flatten to B*T x V
        mask = (Y_flat == self.padding_idx)
        num_tokens = torch.sum(~mask).item()
        Y_hat_correct = Y_hat_flat[range(Y_hat_flat.shape[0]), Y_flat]
        Y_hat_correct[mask] = 0
        ce_loss = -torch.sum(Y_hat_correct) / num_tokens
        return ce_loss

    def train_lm(self, data, print_every=10, num_epochs=1000, batch_size=None, **kwds):
        if batch_size is None:
            batch_size = len(data)    
        opt = torch.optim.Adam(params=self.parameters(), **kwds)
        for i in range(num_epochs):
            opt.zero_grad()
            batch = random.sample(data, batch_size) # shape B x T
            padded_batch = pad_sequences(batch)
            z = self.encode(padded_batch) # shape B x (T+1) x H
            y_hat = self.decode(z) # shape B x (T+1) x V
            #y = torch.roll(padded_batch, -1, -1) # shape B x T
            loss = self.lm_loss(padded_batch, y_hat)
            loss.backward()
            opt.step()
            if i % print_every == 0:
                print("epoch %d, loss = %s" % (i, str(loss.item())), file=sys.stderr)

    def distro_after(self, sequence):
        sequence = list(sequence) + [PADDING_IDX]
        padded = pad_sequences([sequence])
        encoded = self.encode(padded)
        predicted = self.decode(encoded)[0,-1,:]
        return predicted

    def generate(self):
        so_far = []
        while True:
            predicted = self.distro_after(so_far).exp().detach().numpy()
            sampled = np.random.choice(range(len(predicted)), p=predicted)
            yield sampled
            if sampled == self.eos_idx:
                break
            else:
                so_far.append(sampled)

def pad_sequences(xs, padding_idx=PADDING_IDX):
    batch_size = len(xs)
    lengths = [len(x) for x in xs]
    longest = max(lengths)
    padded = torch.ones(batch_size, longest).long() * padding_idx
    for i, length in enumerate(lengths):
        sequence = xs[i]
        padded[i, 0:length] = torch.Tensor(sequence[:length])
    return padded

def example(**kwds):
    """ Minimum possible loss is 1/5 ln 5 = 0.3219 """
    data = [
        [2,2,EOS_IDX],
        [3,3,3,EOS_IDX],
        [4,4,4,4,EOS_IDX],
        [5,5,5,5,5,EOS_IDX],
        [6,6,6,6,6,6,EOS_IDX],
    ]
    vocab_size = len(data) + 2
    lstm = LSTM(vocab_size, vocab_size, 2, 10)
    lstm.train_lm(data, **kwds)
    return lstm

class Indexer:
    def __init__(self, eos_idx=EOS_IDX, padding_idx=PADDING_IDX):
        self.counter = itertools.count()
        self.eos_idx = eos_idx
        self.padding_idx = padding_idx
        self.seen = {EOS: self.eos_idx, PADDING: self.padding_idx}
        
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
    result = [list(map(indexer.index_for, xs)) for xs in xss]
    return result, indexer.seen

def read_unimorph(filename, field=1):
    with open(filename) as infile:
        for line in infile:
            if line.strip():
                parts = line.strip().split("\t")
                yield parts[field].casefold()

def train_unimorph_lm(lang, hidden_size=100, num_layers=2, batch_size=5, **kwds):
    data, vocab = list(format_sequences(read_unimorph("/Users/canjo/data/unimorph/%s" % lang)))
    print("Loaded data for %s..." % lang, file=sys.stderr)
    vocab_size = len(vocab)
    print("Vocab size: %d" % vocab_size, file=sys.stderr)
    lstm = LSTM(vocab_size, vocab_size, num_layers, hidden_size)
    print(lstm, file=sys.stderr)
    lstm.train_lm(data, batch_size=batch_size, num_epochs=10000, print_every=500, **kwds)
    return lstm, vocab

if __name__ == '__main__':
    train_unimorph_lm(*sys.argv[1:])
   
