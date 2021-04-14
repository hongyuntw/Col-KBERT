import torch
import nltk
import spacy
from transformers import BertTokenizerFast
from colbert.modeling.tokenization.utils import _split_into_batches


class QueryTokenizer():
    def __init__(self, query_maxlen):
        self.tok = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.query_maxlen = query_maxlen

        self.Q_marker_token, self.Q_marker_token_id = '[Q]', self.tok.convert_tokens_to_ids('[unused0]')
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        self.mask_token, self.mask_token_id = self.tok.mask_token, self.tok.mask_token_id
        self.nltk_stopwords = nltk.corpus.stopwords.words("english")
    

        assert self.Q_marker_token_id == 1 and self.mask_token_id == 103

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.Q_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix + [self.mask_token] * (self.query_maxlen - (len(lst)+3)) for lst in tokens]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        ids = self.tok(batch_text, add_special_tokens=False)['input_ids']

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.Q_marker_token_id], [self.sep_token_id]
        ids = [prefix + lst + suffix + [self.mask_token_id] * (self.query_maxlen - (len(lst)+3)) for lst in ids]

        return ids

    # when training call here
    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        # add placehold for the [Q] marker
        batch_text = ['. ' + x for x in batch_text]

        obj = self.tok(batch_text, padding='max_length', truncation=True,
                       return_tensors='pt', max_length=self.query_maxlen)

        ids, mask = obj['input_ids'], obj['attention_mask']



        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        ids[ids == 0] = self.mask_token_id


        if bsize:
            batches = _split_into_batches(ids, mask, bsize)
            return batches

        return ids, mask

    # training call here to random mask
    # original version : [MASK] tokens attention is zero
    def tensorize_random_mask(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        # add placehold for the [Q] marker
        batch_text = ['. ' + x for x in batch_text]

        batch_ids = []
        batch_masks = []
        for text in batch_text:
            tokens = self.tok.tokenize(text)
            pos = nltk.pos_tag(tokens)
            new_tokens = []
            new_masks = []
            for i in range(len(tokens)):
                p = pos[i]
                token = tokens[i]
                # if token is Noun or adj and is not stop words and not subword token
                if ('NN' in p[1] or 'JJ' in p[1]) and (token not in self.nltk_stopwords) and '#' not in token: 
                    new_tokens.append(token)
                    new_tokens.append('[MASK]')
                    new_masks.append(1)
                    # add 1 means let BERT do self attention with mask token, and zero is not
                    # 不讓bert做self attention代表不讓他破壞句法結構? 最後一層layer再猜
                    new_masks.append(0)
                else:
                    new_tokens.append(token)
                    new_masks.append(1)

            # padding or truncate
            if len(new_tokens) >= self.query_maxlen - 2:
                new_tokens = ['[CLS]'] + new_tokens[:self.query_maxlen - 2] + ['[SEP]']
                new_masks = [1] + new_masks[:self.query_maxlen - 2] + [0]
            else:
                new_tokens = ['[CLS]'] +  new_tokens + ['[MASK]'] * (self.query_maxlen - 2 - len(new_tokens)) + ['[SEP]']
                new_masks = new_masks + [0] * (self.query_maxlen - len(new_masks))
            new_ids = self.tok.convert_tokens_to_ids(new_tokens)
            new_ids[1] = self.Q_marker_token_id

            # print(text)
            # print(new_tokens , len(new_tokens))
            # print(new_ids , len(new_ids))
            # print(new_masks , len(new_masks))

            batch_ids.append(new_ids)
            batch_masks.append(new_masks)

        ids = torch.tensor(batch_ids)
        mask = torch.tensor(batch_masks)

        if bsize:
            batches = _split_into_batches(ids, mask, bsize)
            return batches

        return ids, mask

    