# coding: utf-8
import json
import argparse
import math
import torch
import torch.nn as nn
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
        self.dict_path = os.path.join(path, 'dict.json')
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
# - inference function
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--save', type=str, default=model_path, help='path to save the model')
args = parser.parse_args()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
corpus = Corpus(args.data, tokenize=False)

tokenizer_len = len(corpus.dictionary)
print('tokenizer length: ', tokenizer_len)


with open(args.save, 'rb') as f:
    model = torch.load(f)


def inference(number_list):
    model.eval()
    batch_inputs = []
    batch_y_inputs = []

    for number in number_list:
        number_str = str(number)
        words = [i for i in number_str]
        words = ['<sos>'] + words + ['<eos>']
        words.extend(['<pad>'] * (max_seq_length - len(words)))
        ids = []
        for word in words:
            ids.append(corpus.dictionary.word2idx[word])
        batch_inputs.append(ids)
        batch_y_inputs.append([corpus.dictionary.word2idx['<sos>']])

    batch_inputs = torch.tensor(batch_inputs).to(device)
    batch_y_inputs = torch.tensor(batch_y_inputs).to(device)

    results = []
    with torch.no_grad():
        for i in range(0, max_seq_length):  # max_seq_length
            input_seq_length = batch_y_inputs.size(1)
            tgt_mask = model.get_tgt_mask(input_seq_length).to(device)

            output = model(batch_inputs, batch_y_inputs, tgt_mask=tgt_mask)
            output = output[:, -1:, :]
            output_ids = torch.argmax(output, -1)

            batch_y_inputs = torch.cat([batch_y_inputs, output_ids], dim=1)

            output_texts = corpus.decode(output_ids)
            print(output_texts)
            results.append(output_texts)

    print('Result: ')
    for i in range(0, len(number_list)):
        word_list = [results[j][i][0] for j in range(0, len(results))]
        word_list = [i for i in word_list if i not in ['<eos>', '<sos>', '<pad>']]
        words = ' '.join(word_list)
        print(number_list[i], '-->', words)
    return results


inference([
    1235678,
    200.3236,
    -2000,
    10000001,
    66666666,
    -823982502.002,
    987654321.12,
    3295799.9873462
])
