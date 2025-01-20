from transformers import RobertaTokenizerFast
from datasets import IterableDataset
from transformers import RobertaTokenizer
import itertools


class LineByLineTextDataset(IterableDataset):

    def __init__(self, tokenizer, file_path, max_length=512, offsets=None):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.max_length = max_length
        self.offsets = offsets
        # Check if using RoBERTa/CodeBERT tokenizer
        self.is_roberta = isinstance(self.tokenizer, (RobertaTokenizer, RobertaTokenizerFast))
        # Check if using CodeBERT
        self.is_codebert = "codebert" in tokenizer.name_or_path.lower()
        with open(file_path, encoding="utf-8") as f:
            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        return self.process_line(i, self.lines[i])

    def process_line(self, worker_id, line):
        if len(line) == 0 or line.isspace() or not len(line.split(' ||| ')) == 2:
            return None

        sent_src, sent_tgt = line.strip().split(' ||| ')
        sent_src = sent_src.strip().split()
        sent_tgt = sent_tgt.strip().split()

        if self.is_roberta:
            # CodeBERT/RoBERTa specific tokenization
            token_src = [self.tokenizer.tokenize(" " + word if i > 0 else word) 
                        for i, word in enumerate(sent_src)]
            token_tgt = [self.tokenizer.tokenize(" " + word if i > 0 else word) 
                        for i, word in enumerate(sent_tgt)]
        else:
            # Original BERT tokenization
            token_src = [self.tokenizer.tokenize(word) for word in sent_src]
            token_tgt = [self.tokenizer.tokenize(word) for word in sent_tgt]

        # Convert tokens to ids
        wid_src = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src]
        wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

        # Prepare for model with special tokens and truncation
        ids_src = self.tokenizer.prepare_for_model(
            list(itertools.chain(*wid_src)),
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding=False
        )['input_ids']

        ids_tgt = self.tokenizer.prepare_for_model(
            list(itertools.chain(*wid_tgt)),
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding=False
        )['input_ids']

        # Create word mapping for subword tokens
        bpe2word_map_src = []
        for i, word_list in enumerate(token_src):
            bpe2word_map_src += [i for _ in word_list]

        bpe2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            bpe2word_map_tgt += [i for _ in word_list]

        if ids_src.shape[1] <= 2 or ids_tgt.shape[1] <= 2:
            return None

        return (
            worker_id,
            ids_src.squeeze(0),
            ids_tgt.squeeze(0),
            bpe2word_map_src,
            bpe2word_map_tgt,
            sent_src,
            sent_tgt
        )