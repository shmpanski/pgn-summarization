import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import masked_cross_entropy


class PGNEncoder(nn.Module):
    r"""Encoder for pointer-generator nn

        Args:
            hidden_size (int): The number of features in hidden state
            n_layers (int, optional): Number of recurrent layers. Default: 1
            dropout (float, optional): If non-zero, introduces a `Dropout` layer on the outputs of each
              GRU layer except the last layer, with dropout probability equal to :attr:`dropout`. Default: 0

        Inputs: input, input_lens, h_0
            - **input** of shape `(seq_len, batch, hidden_size)`: tensor containing embedded words
              of the input sequence.
            - **input_lens** list of lens of input sequences.
            - **h_0** of shape `(n_layers * 2, batch, hidden_size)`: tensor
              containing the initial hidden state for each element in the batch.
              Defaults to zero if not provided.

        Outputs: output, h_n
            - **output** of shape `(seq_len, batch, hidden_size)`: tensor
              containing the output features h_t for each t.
            - **h_n** of shape `(n_layers * 2, batch, hidden_size)`: tensor
              containing the hidden state for `t = seq_len`
        """

    def __init__(self, vocabulary_size, hidden_size, n_layers=1, dropout=0):
        super(PGNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(dropout if n_layers != 1 else 0), bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, input_lens, h_0=None):
        packed = nn.utils.rnn.pack_padded_sequence(input, input_lens)

        output, h_n = self.gru(packed, h_0)
        output, output_lens = torch.nn.utils.rnn.pad_packed_sequence(output)
        output = self.dropout(output)

        # Sum bidirectional
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]

        return output, h_n


class PGNDecoder(nn.Module):
    r"""Decoder for pointer-generator nn

        Args:
            vocabulary_size (int): The size of the vocabulary of embeddings
            hidden_size (int): The number of features in hidden state
            n_layers (int, optional): Number of recurrent layers. Default: 1
            dropout (float, optional): If non-zero, introduces a `Dropout` layer on the outputs of each
              GRU layer except the last layer, with dropout probability equal to :attr:`dropout`. Default: 0

        Inputs: input, encoder_states, oov_enc_mask, hidden
            - **input** of shape `(1, batch, hidden_size)`: tensor containing previous embedding of word
              from generated sequence.
            - **encoder_states** of shape `(seq_len, batch, hidden_size)`: tensor
              containing features of encoded input sequence
            - **oov_enc_mask** of shape `(seq_len, batch)`: tensor containing binary
              mask of out of vocabulary words. 1 means oov word, 0 means word from vocabulary.
            - **hidden** of shape `(n_layers, batch, hidden_size)`: tensor
              containing the previous hidden state of generated word.
              Defaults to zero if not provided.

        Outputs: extended_vocab_distr, hidden, p, attention_distr
            - **extended_vocab_distr** of shape `(batch, vocabulary_size + seq_len)`: tensor
              containing vocabulary distribution for generated word.
            - **hidden** of shape `(n_layers, batch, hidden_size)`: tensor
              containing the hidden state for previously generated word.
            - **p** of shape `(batch)`: tensor containing scalar probabilities for generating or
              pointing word.
            - **attention_distr** of shape `(seq_len, batch)`: tensor containing attention
              distribution along encoder hidden states.
        """

    def __init__(self, vocabulary_size, hidden_size, n_layers=1, dropout=0):
        super(PGNDecoder, self).__init__()
        self.hidden_size = hidden_size

        # Decoder
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(dropout if n_layers != 1 else 0))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size * 2, vocabulary_size)

        # Attention
        self.v = nn.Parameter(torch.FloatTensor(hidden_size, 1))
        self.Wh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self.Ws = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.FloatTensor(1, hidden_size))

        # Pointer
        self.pointer_Wh = nn.Linear(hidden_size, 1, False)
        self.pointer_Ws = nn.Linear(hidden_size, 1, False)
        self.pointer_Wx = nn.Linear(hidden_size, 1, True)

    def forward(self, input, encoder_states, oov_enc_mask, hidden=None):
        decoder_state, hidden = self.gru(input, hidden)
        decoder_state = self.dropout(decoder_state)

        # Calculate attention weights
        attention_distr = self.attention(encoder_states, decoder_state).transpose(0, 1)
        encoder_states = encoder_states.transpose(0, 1)
        context = attention_distr.unsqueeze(1).bmm(encoder_states).squeeze(1)
        global_state = torch.cat((decoder_state.squeeze(0), context), 1)

        # Calculate generation probability
        p = F.sigmoid(self.pointer_Wh(context) + self.pointer_Ws(decoder_state) + self.pointer_Wx(input.squeeze(0)))
        p = p.squeeze(0)
        vocab_distr = p * F.selu(self.out(global_state))
        oov_vocab_distr = (1 - p) * (attention_distr * oov_enc_mask.float().transpose(0, 1))
        extended_vocab_distr = torch.cat((vocab_distr, oov_vocab_distr), 1)

        return extended_vocab_distr, hidden, p.squeeze(), attention_distr.transpose(0, 1)

    def attention(self, enc_states, dec_state):
        # Attention like in Bahdanau et al. (2015)
        x1 = enc_states.matmul(self.Wh)
        x2 = dec_state.expand_as(enc_states).matmul(self.Ws)
        bias = self.b.unsqueeze(0).expand_as(enc_states)
        res = F.tanh(x1 + x2 + bias).matmul(self.v).squeeze(2)

        return F.softmax(res, 0)


class PGNN(nn.Module):
    r"""Pointer-generator NN

        Args:
            vocabulary_size (int): The size of the vocabulary of embeddings
            hidden_size (int): The number of features in hidden state
            n_layers (int, optional): Number of recurrent layers. Default: 1
            dropout (float, optional): If non-zero, introduces a `Dropout` layer on the outputs of each
              GRU layer except the last layer, with dropout probability equal to :attr:`dropout`. Default: 0

        Inputs: input_seq, input_lens, oov_mask, max_target_len, target_seq
            - **input_seq** long tensor of shape `(seq_len, batch)`: tensor containing encoded words
              of the input sequence.
            - **input_lens** list of lens of input sequences.
            - **oov_mask** tensor of shape `(seq_len, batch)`: tensor containing binary
              mask of out of vocabulary words. 1 means oov word, 0 means word from vocabulary.
            - **max_target_len** scalar, max length of target sequence
            - **target_seq** tensor of shape `(target_seq_len, batch)`. If provided, decoder will use
              words from target sequence to generate next word. If don't provided, decoder will use previously
              generated words.

        Outputs: target_seq_distr if **target_seq** provided,
                 else (target_seq, target_seq_distr, pointer_distr, att_distr)
            - **target_seq_distr** of shape `(max_target_len, batch, vocabulary_size + input_seq_len)`: tensor
              containing vocabulary distribution on generated sequence.

            - **target_seq** of shape `(max_target_len, batch)`: tensor, containing encoded generated sequence.
            - **pointer_distr** of shape `(max_target_len, batch)`: tensor, containing pointer probabilities.
            - **att_distr** of shape `(max_target_len, seq_len, batch)`: tensor, containing attention distributions
              for each generated word.
        """

    def __init__(self, vocabulary_size, hidden_size, n_layers=1, dropout=0):
        super(PGNN, self).__init__()

        self.vocabulary_size = vocabulary_size

        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.encoder = PGNEncoder(vocabulary_size, hidden_size, n_layers, dropout)
        self.decoder = PGNDecoder(vocabulary_size, hidden_size, n_layers, dropout)

    def forward(self, input_seq, input_lens, oov_mask, max_target_len, target_seq=None):
        input_emb_seq = self.embedding(input_seq)
        # Initial word -- <sos> tag, encoded by `2`
        initial_word = (torch.ones(1, input_seq.shape[1], device=input_seq.device) * 2).long()
        initial_emb_word = self.embedding(initial_word)

        encoder_state, _ = self.encoder(input_emb_seq, input_lens)
        dec_hidden = None

        if target_seq is not None:
            generated_seq = []
            for i in range(max_target_len):
                ext_voc_distr, dec_hidden, _, _ = self.decoder(initial_emb_word, encoder_state, oov_mask, dec_hidden)
                generated_seq.append(ext_voc_distr)
                initial_emb_word = self.embedding(target_seq[i].unsqueeze(0))

            return torch.stack(generated_seq)
        else:
            generated_seq, generated_seq_distr, pointer_prob, attention_distr = [], [], [], []
            for i in range(max_target_len):
                ext_voc_distr, dec_hidden, p, att = self.decoder(initial_emb_word, encoder_state, oov_mask, dec_hidden)

                word_idx = self.word_selector(ext_voc_distr)

                generated_seq.append(word_idx)
                generated_seq_distr.append(ext_voc_distr)
                pointer_prob.append(p)
                attention_distr.append(att)

                initial_emb_word = self.embedding(self.word_selector(ext_voc_distr, False, self.vocabulary_size).unsqueeze(0))
            return torch.stack(generated_seq), torch.stack(generated_seq_distr), torch.stack(pointer_prob), torch.stack(
                attention_distr)

    def train_step(self, optimizer, batch, grad_clip):
        r"""Take a step of learning process

            Args: optimizer, batch, grad_clip
                - **optimizer** optimizer for learning
                - **batch** torchtext.data.batch.Batch object of TabularDataset
                - **grad_clip** value of gradient clipping

            Output: loss
                - **loss** loss value
        """
        optimizer.zero_grad()

        input_seq, input_lens, target_seq, target_lens = *batch.src, *batch.trg
        # Extended target is a target, encoded using OOV encoding
        target_seq_ext, _ = batch.trg_ext
        max_target_len = target_lens.max().item()

        # Calculate out of vocabulary mask. If word has <unk> tag, this is oov.
        oov_mask = (input_seq == 0).long().to(input_seq.device)

        out_seq_distr = self.forward(input_seq, input_lens, oov_mask, max_target_len, target_seq)

        loss = masked_cross_entropy(out_seq_distr.transpose(0, 1).contiguous(),
                                    target_seq_ext.transpose(0, 1).contiguous(), target_lens)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        optimizer.step()

        return loss.item(), grad_norm

    def evaluate(self, batch):
        r""" Take a step of learning process

                Args: batch, grad_clip
                    - **batch** torchtext.data.batch.Batch object of TabularDataset
                    - **grad_clip** value of gradient clipping
                Output: loss, target_seq, target_seq_distr, pointer_distr, att_distr
                    - **loss** loss value
                    - **target_seq** of shape `(max_target_len, batch)`: tensor, containing encoded generated sequence.
                    - **target_seq_distr** of shape `(max_target_len, batch, vocabulary_size + input_seq_len)`: tensor
                      containing vocabulary distribution on generated sequence.
                    - **pointer_distr** of shape `(max_target_len, batch)`: tensor, containing pointer probabilities.
                    - **att_distr** of shape `(max_target_len, seq_len, batch)`: tensor, containing attention distributions
                      for each generated word.
                """
        self.eval()

        input_seq, input_lens, target_seq, target_lens = *batch.src, *batch.trg
        target_seq_ext, _ = batch.trg_ext
        max_target_len = target_lens.max().item()

        # Calculate out of vocabulary mask. If word has <unk> tag, this is oov.
        oov_mask = input_seq == 0

        target_seq, target_seq_distr, pointer_distr, att_distr = self.forward(input_seq, input_lens,
                                                                              oov_mask, max_target_len)
        loss = masked_cross_entropy(target_seq_distr.transpose(0, 1).contiguous(),
                                    target_seq_ext.transpose(0, 1).contiguous(), target_lens)
        self.train()

        return loss.item(), target_seq, target_seq_distr, pointer_distr, att_distr

    def word_selector(self, distribution, use_oov=True, vocabulary_size=None):
        args = torch.argmax(distribution, 1).long()
        if not use_oov:
            assert vocabulary_size is not None, "vocabulary size doesn't defined"
            # Be careful. Here we decode words, get their indices on vocabulary. If word isn't in vocabulary,
            # we decode it as 0, what means <unk> tag in DataLoader.
            return args * (args < vocabulary_size).long()
        else:
            return args
