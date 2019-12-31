from pathlib import Path
from tqdm import tqdm
import logging
import csv
import collections
import numpy as np

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, labels=None, doc_span_index = None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels
        self.doc_span_index = doc_span_index


class InputFeatures(object):
    def __init__(self,guid, input_ids, input_mask, segment_ids, label_ids, doc_span_index):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.doc_span_index = doc_span_index
        
class DataProcessor(object):

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    
    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError() 

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

        
def _read_tsv(input_file, cls = "\t", quotechar=None):
    reader = csv.reader(input_file.open('r'), delimiter=cls, quotechar=None)
    lines = [line for line in reader]
    return lines

class MultiClassTextProcessor(DataProcessor):

    def __init__(self, data_path, train_file, test_file,labels = None, dev_file = None):
        self.train_path = data_path/train_file
        self.test_path = data_path/test_file
        self.train_file = _read_tsv(data_path/train_file)
        self.test_file = _read_tsv(data_path/test_file)
        if dev_file!= None:
            self.dev_path = data_path/dev_file
            self.dev_file = _read_tsv(data_path/dev_file)
        self.labels = labels
    
    def get_train_examples(self):        
        logger.info("LOOKING AT {}".format(self.train_path))
        return self._create_examples(self.train_file, "train")
        
    def get_dev_examples(self):
        logger.info("LOOKING AT {}".format(self.dev_path))
        if self.dev_file!= None:
            return self._create_examples(self.dev_file, "dev")
        else:
            raise ValueError('There is no dev file')
    
    def get_test_examples(self):
        logger.info("LOOKING AT {}".format(self.test_path))
        return self._create_examples(self.test_file, "test")

    def get_labels(self):
        """See base class."""
        if self.labels == None:
            self.labels = ['0', '1']
        return self.labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = None
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, labels=label))
        return examples
    
    
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, doc_stride):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label : i for i, label in enumerate(label_list)}
    features_all = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None

        max_tokens_for_doc = max_seq_length  - 2
        _DocSpan = collections.namedtuple(  
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0

        while start_offset < len(tokens_a):
            length = len(tokens_a) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(tokens_a):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            features = []
            tokens = []
            segment_ids = [0]*max_seq_length
            tokens.append("[CLS]")
            
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                tokens.append(tokens_a[split_token_index])

            tokens.append("[SEP]")

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding

            
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            labels_ids = []
            for label in example.labels:
                labels_ids.append(float(label))


            if ex_index < 10:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(
                        [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: %s (id = %s)" % (example.labels, labels_ids))

            features.append(
                    InputFeatures(guid = example.guid,
                                  input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_ids=labels_ids, 
                                  doc_span_index = doc_span_index))
            features_all.extend(features)## extend가 맞남... 모르겟다링~
    return features_all


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()