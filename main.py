# coding: utf-8
import json
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import os


# set this accordingly
max_seq_length = 32
model_path = 'd:/data/models/one_transformer_model.pt'


###############################################################################
# Part I
# - define the Tokenizer class, Dictionary, Corpus
# - define the Model class
###############################################################################


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, tokenize=True):
        self.dictionary = Dictionary()
        self.dict_path = './data/dict.json'
        self.train_data = os.path.join(path, 'train.tsv')
        self.valid_data = os.path.join(path, 'valid.tsv')
        self.test_data = os.path.join(path, 'test.tsv')
        self.init_dictionary(self.train_data)

        if tokenize:
            self.train = self.tokenize(self.train_data)
            self.valid = self.tokenize(self.valid_data)
            self.test = self.tokenize(self.test_data)

    def init_dictionary(self, data_path):
        if os.path.exists(self.dict_path):
            print('loaded dict from local file: {0}'.format(self.dict_path))
            self.load_dictionary()
        else:
            print('build dict from training corpus file: {0}'.format(data_path))
            self.dictionary.add_word('<sos>')
            self.dictionary.add_word('<eos>')
            self.dictionary.add_word('<pad>')
            max_len = 0
            with open(data_path, 'r', encoding="utf8") as f:
                for line in f:
                    segs = line.strip().split('\t')
                    for seg in segs:
                        words = seg.split()
                        if len(words) > max_len:
                            max_len = len(words)
                        for word in words:
                            self.dictionary.add_word(word)
            print('length of dictionary: ', len(self.dictionary))
            print('max sentence length: ', max_len)

    def save_dictionary(self):
        with open(self.dict_path, 'w', encoding='utf-8') as wf:
            wf.write(json.dumps(self.dictionary.idx2word, ensure_ascii=False, indent=4))

    def load_dictionary(self):
        with open(self.dict_path, 'r', encoding='utf-8') as rf:
            word_list = json.load(rf)
            self.dictionary = Dictionary()
            for word in word_list:
                self.dictionary.add_word(word)

    def decode(self, batch_ids):
        batch_results = []
        for batch in batch_ids:
            tokens = [self.dictionary.idx2word[i] for i in batch]
            batch_results.append(tokens)
        return batch_results

    def tokenize(self, path):
        assert os.path.exists(path)
        with open(path, 'r', encoding="utf8") as f:
            samples = []
            for line in f:
                line = line.strip()
                columns = line.split('\t')
                sample_ids = []
                for column in columns:
                    words = ['<sos>'] + column.split() + ['<eos>']
                    words.extend(['<pad>'] * (max_seq_length - len(words)))
                    ids = []
                    for word in words:
                        ids.append(self.dictionary.word2idx[word])
                    sample_ids.append(ids)
                samples.append(sample_ids)
        data = torch.tensor(samples).type(torch.int64)
        return data


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    the transformer architecture model contains an embedding module, a pytorch implemented Transformer, and an output
    linear layer.
    You can regard it as a wrapper class for the Transformer

    Some notes:
        The d_model is one of the worst variable namings I've ever seen. this is actually the embedding size.
        I prefer batch_first set to True, this is more intuitive from my understanding.
    """
    def __init__(self, tokenizer_len, d_model, nhead, dim_feedforward, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead, dim_feedforward=dim_feedforward,
                                          num_encoder_layers=nlayers,
                                          batch_first=True)
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.embedding = nn.Embedding(tokenizer_len, d_model)
        self.d_model = d_model
        self.out = nn.Linear(d_model, tokenizer_len)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        nn.init.uniform_(self.embedding.weight, -init_range, init_range)
        nn.init.uniform_(self.out.weight, -init_range, init_range)
        nn.init.zeros_(self.out.bias)

    def forward(self, src, tgt, tgt_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)

        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        # here set only the tgt_mask, if not use parameter name, the 3rd parameter is src_mask
        transform_output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        output = self.out(transform_output)
        return output

    def get_tgt_mask(self, size):
        # Generates a square matrix where each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask


###############################################################################
# Part II
# - define the training parameters
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/10m', help='location of the data corpus')
parser.add_argument('--d_model', type=int, default=256, help='size of word embeddings')
parser.add_argument('--dim_feedforward', type=int, default=1024, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=4, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
parser.add_argument('--seq_len', type=int, default=32, help='sequence length')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default=model_path, help='path to save the model')
parser.add_argument('--nhead', type=int, default=2, help='number of heads in the transformer encoder/decoder')
parser.add_argument('--dry-run', action='store_true', help='verify the code and the model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


###############################################################################
# Part III
# prepare the data
# prepare the model and other variables
###############################################################################


def batchify(data, batch_size):
    """
    data shape is [n, 2, sequence_len]
    output shape is [batch_count, batch_size, 2, sequence_len]
    """
    total_batch = data.size(0) // batch_size
    batch_list = []
    for i in range(0, total_batch):
        # yield data[i * batch_size: (i + 1) * batch_size]  # need handle multiple times.
        batch_list.append(data[i * batch_size: (i + 1) * batch_size])
    return batch_list


def get_batch(data, i, batch_size):
    """
    get a batch from the batchify data
    """
    target = data[i: i + batch_size]
    return data, target


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


corpus = Corpus(args.data)
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, args.batch_size)
test_data = batchify(corpus.test, args.batch_size)
tokenizer_length = len(corpus.dictionary)
print('tokenizer length: ', tokenizer_length)
model = TransformerModel(tokenizer_length,
                         args.d_model,
                         args.nhead,
                         args.dim_feedforward,
                         args.nlayers,
                         args.dropout).to(device)
parameter_count = count_parameters(model)
print('total parameters: ', parameter_count)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)


###############################################################################
# Part IV
# - training function and evaluation function
# - evaluation function
###############################################################################

def train():
    """
    training function for an epoch
    """
    model.train()
    total_loss = 0.
    start_time = time.time()

    index = 0
    total_batch = len(train_data)
    for batch in train_data:
        optimizer.zero_grad()
        X, y = batch[:, 0, :].to(device), batch[:, 1, :].to(device)
        y_input = y[:, :-1]
        y_expected = y[:, 1:]
        input_seq_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(input_seq_length).to(device)
        # model.zero_grad()
        output = model(X, y_input, tgt_mask)
        output = output.permute(0, 2, 1)

        loss = criterion(output, y_expected)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if index % args.log_interval == 0 and index > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('\nprogress: {0}/{1}, total loss: {2}, current loss: {3}, elapsed: {4}'.format(
                index,
                total_batch,
                total_loss,
                cur_loss,
                round(elapsed, 1)))
            total_loss = 0
            start_time = time.time()
        index += 1


def evaluate(data_source):
    """
    evaluation function
    """
    model.eval()
    total_loss = 0.
    total_batch = 0
    with torch.no_grad():
        for batch in data_source:
            X, y = batch[:, 0, :].to(device), batch[:, 1, :].to(device)
            y_input = y[:, :-1]
            y_expected = y[:, 1:]
            input_seq_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(input_seq_length).to(device)
            model.zero_grad()

            output = model(X, y_input, tgt_mask=tgt_mask)
            output = output.permute(0, 2, 1)

            loss = criterion(output, y_expected)
            total_loss += loss.detach().item()
            total_batch += 1

    return total_loss / total_batch


print('\n\nStart training, total epochs: {0}\n'.format(args.epochs))
best_val_loss = None
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('\nEpoch ', epoch,  '-' * 89)
        print(' - evaluation loss: {0}'.format(val_loss))
        print('Epoch ', epoch,  '-' * 89)
        with open(args.save, 'wb') as f:
            torch.save(model, f)
            corpus.save_dictionary()  # save the tokenizer dictionary

        # Save the model if the validation loss is the best so far.
        # if not best_val_loss or val_loss < best_val_loss:
        #     with open(args.save, 'wb') as f:
        #         torch.save(model, f)
        #         corpus.save_dictionary()  # save the tokenizer dictionary
        #     best_val_loss = val_loss
        # update learning rate based on lr scheduler
        lr_scheduler.step()
except KeyboardInterrupt:
    print('Exiting training!')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

test_loss = evaluate(test_data)
print('\nTesting ',  '-' * 89)
print('Test loss {:5.2f}, test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('\nTesting ',  '-' * 89)
