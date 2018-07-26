import csv
import itertools
import re
from collections import Counter, defaultdict
from utils import bcolors

import numpy as np
import torch as t


def default_tokenizer(token_pattern=r"(?u)\b[a-zA-Zа-яА-Я]{1,}\b"):
    token_pattern = re.compile(token_pattern)

    return lambda doc: token_pattern.findall(doc)


class DataLoader:
    def __init__(self, directory, parts, max_vocab_size, tokenizer=default_tokenizer()):
        """
        Data loader for sentence summarization dataset

        Attributes: global_itos, global_stoi, vocab_size, data_parts, parts_lens
            - **global_itos** decoder list of words
            - **global_stoi** encoder dictionary
            - **vocab_size** vocabulary size
            - **data_parts** dictionary of parts, each value is list of examples from dataset part
            - **data_lens** dictionary of parts, each value is length of dataset part

        Inputs: directory, parts, max_vocab_size, tokenizer
            - **directory** dataset directory
            - **parts** list of string, names of each part of dataset
            - **max_vocab_size** maximum vocabulary size
            - **tokenizer** tokenizer function. Default data.default_tokenizer()
        """
        self.directory = directory
        self.max_vocab_size = max_vocab_size
        self.tokenizer = tokenizer

        self.specials = ['<unk>', '<pad>', '<sos>', '<eos>']
        self.unk_tag, self.pad_tag, self.sos_tag, self.eos_tag = self.specials
        self.unk_idx, self.pad_idx, self.sos_idx, self.eos_idx = range(4)
        self.global_itos, self.global_stoi = None, None
        self.vocab_size = 4
        self.data_parts, self.part_lens = None, None

        self.preprocess(parts)

    def preprocess(self, parts):
        """
        Preprocess dataset

        Args: parts
            - **parts** list [string] names of dataset parts. For eample: train, test, validate, sample
        """
        self.data_parts = {part: list(self.from_csv(part)) for part in parts}
        self.part_lens = {part: len(self.data_parts[part]) for part in parts}

        merged_data_parts = itertools.chain(*list(self.data_parts.values()))

        self.global_itos, self.global_stoi = self.build_dictionary(merged_data_parts)
        self.vocab_size = len(self.global_itos)

    def build_dictionary(self, data):
        """
        Create encoder dictionary and decoder list

        Args: data
            - **data** iterable (string, string) examples from whole dataset

        Output itos, stoi
            - **itos** decoder list of words
            - **stoi** encoder dictionary
        """
        itos = self.specials.copy()

        counter = Counter()
        for example in data:
            counter.update(example[0])
            counter.update(example[1])

        frequencies = counter.most_common(self.max_vocab_size - len(self.specials))
        itos.extend(x[0] for x in frequencies)
        stoi = defaultdict(self._get_unk_idx(), [(word, i) for i, word in enumerate(itos)])

        return itos, stoi

    def from_csv(self, part):
        """
        Read and tokenize data. This loader works only with TSV files.

        Parameters: part
            - **part**: string name of part of dataset

        Output:
            generator of pairs ([string], [string]) - tokenized examples
        """
        filename = self.directory + part + '.tsv'
        with open(filename) as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                yield self.tokenizer(row[0].lower()), self.tokenizer(row[1].lower())

    def process(self, data, device):
        """
        Pad and numericalize batch

        Args: data, device
            - **data** list [[string]] of words from text
            - **device** torch device

        Output data, lengths
            - **data** tensor (max_seq_len, batch)
            - **lenghts** tensor (batch) containing length of each sequence
        """

        padded, lengths = self.pad(data)
        return self.numericalize(padded, lengths, self.global_stoi, device)

    def pad(self, data):
        """
        Add <sos>, <eos> tags and pad sequences from batch

        Args: data
            - **data** list [[string]] of words from text

        Output data, lengths
            - **data** padded list [[string]] of sizes (batch, max_seq_len + 2)
        """
        data = list(map(lambda x: [self.sos_tag] + x + [self.eos_tag], data))
        lens = [len(s) for s in data]
        max_len = max(lens)
        for i, length in enumerate(lens):
            to_add = max_len - length
            data[i] += [self.pad_tag] * to_add
        return data, lens

    def numericalize(self, data, lengths, stoi, device):
        """
        Numericalize batch

        Args: data, lengths, stoi, device
            - **data** list [[string]] of words from padded text
            - **lengths** list [int] containing list of sequences lengths
            - **stoi** dictionary mapping words to indexes
            - **device** torch device

        Output data, lengths
            - **data** tensor (max_seq_len, batch)
            - **lenghts** tensor (batch) containing length of each sequence
        """
        nums = [[stoi[word] for word in ex] for ex in data]
        lens_tensor = t.tensor(lengths, dtype=t.long, device=device)
        nums_tensor = t.tensor(nums, dtype=t.long, device=device)
        nums_tensor = nums_tensor.t_()
        return nums_tensor, lens_tensor

    def next_batch(self, part, batch_size, device):
        """
        Get next batch

        Args: part, batch_size, device
            - **part** string, part of the dataset
            - **batch_size** int batch size
            - **device** torch.device
        Outputs: Batch batch
            - **batch** Batch object with attributes src, trg, trg_ext.
              Each attribute is a tensor of shape (seq_len, batch_size) containing encoded sequences.
        """
        indexes = np.random.randint(0, self.part_lens[part], batch_size)
        batch = [self.data_parts[part][i] for i in indexes]
        return Batch(sorted(batch, key=lambda x: len(x[0]), reverse=True), self, device)

    def _get_unk_idx(self):
        return lambda: self.unk_idx


class Batch:
    def __init__(self, data, loader, device):
        """
        Batch representation

        Attributes: src, trg, trg_ext
            - **src**: (Tensor, Tensor) tuple of source batch tensor and corresponding length tensor
            - **trg**: (Tensor, Tensor) tuple of target batch tensor and corresponding length tensor
            - **trg_ext**: (Tensor, Tensor) tuple of target batch, encoded with rules, applied to
              OOV words. If word isn't in global vocabulary, it's index would be position of corresponding word from
              input sequence, else 0

        Inputs: data, loader, device
            - **data** list of pairs (src, trg) texts, where src and trg are lists of splitted sequences.
            - **loader** DataLoader object of dataset
            - **device** torch.device
        """
        self.loader = loader
        self.device = device

        src = [ex[0] for ex in data]
        trg = [ex[1] for ex in data]

        setattr(self, 'src', loader.process(src, device))
        setattr(self, 'trg', loader.process(trg, device))

        src_padded, src_lens = loader.pad(src)
        trg_padded, trg_lens = loader.pad(trg)
        self.local_itos, self.local_stoi = self.build_local_dictionary(src_padded)
        trg_numericalized = self.numericalize(trg_padded, self.local_stoi, device)

        oov_mask = (self.trg[0] == 0).long()
        unknown_mask = (trg_numericalized == 0).long()
        inv_unknown_mask = 1 - unknown_mask

        trg_extended = self.trg[0] + (oov_mask * (trg_numericalized + loader.vocab_size - 1) * inv_unknown_mask)

        setattr(self, 'trg_ext', (trg_extended, trg_lens))

    def build_local_dictionary(self, data):
        itos = [[word for word in e] for e in data]
        stoi = [defaultdict(lambda: 0, [(tok, i) for i, tok in enumerate(e, 1)]) for e in data]
        return itos, stoi

    def numericalize(self, data, stoi, device):
        nums = [[stoi[i][word] for word in ex] for i, ex in enumerate(data)]
        nums_tensor = t.tensor(nums, dtype=t.long, device=device)
        nums_tensor = nums_tensor.t_()
        return nums_tensor

    def decode(self, batch, console_colorize=False):
        """
        Decode encoded batch of sequences

        Inputs: batch
            - **batch** tensor of shape (seq_len, batch_size)
        """
        return [[self.decode_word(word, batch_id, console_colorize) for word in seq] for batch_id, seq in enumerate(batch.t_())]

    def decode_word(self, idx, batch_id, console_colorize=False):
        if idx < self.loader.vocab_size:
            word = self.loader.global_itos[idx]
        else:
            word = self.local_itos[batch_id][idx - self.loader.vocab_size]
            word = bcolors.OKGREEN + word + bcolors.ENDC if console_colorize else word
        return word
