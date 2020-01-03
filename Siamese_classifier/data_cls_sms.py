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
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, labels=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        
        
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
    
    
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        labels_ids = []
        for label in example.labels:
            labels_ids.append(float(label))

#         label_id = label_map[example.label]
        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.labels, labels_ids))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=labels_ids))
    return features



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
            
            
            
class SiameseFeatures(object):
    """A single set of features of data."""
    def __init__(self,feature_id, data_a, data_a_mask, data_a_lab,
                 data_b, data_b_mask, data_b_lab, label):
        self.feature_id = feature_id
        self.data_a = data_a
        self.data_a_mask = data_a_mask
        self.data_a_lab = data_a_lab
        self.data_b = data_b
        self.data_b_mask = data_b_mask
        self.data_b_lab = data_b_lab
        self.label = label

def convert_to_siamese_features(tf):
    siamese_features = []
    tf_len = len(tf)
    for idx in range(tf_len):
        data_a  = tf[idx].input_ids
        data_a_mask = tf[idx].input_mask
        data_a_lab = tf[idx].label_ids
        for idx_2 in range(idx, tf_len):
            data_b = tf[idx_2].input_ids
            data_b_mask = tf[idx_2].input_mask
            data_b_lab = tf[idx_2].label_ids
            label = int(np.where(tf[idx].label_ids == tf[idx_2].label_ids, 
                             1,0))
            siamese_features.append(SiameseFeatures(feature_id = 'train_{}_{}'.format(int(idx), int(idx_2)), 
                                                    data_a = data_a, 
                                                    data_a_mask = data_a_mask, 
                                                    data_a_lab = data_a_lab, 
                                                    data_b = data_b, 
                                                    data_b_mask = data_b_mask, 
                                                    data_b_lab = data_b_lab, 
                                                    label = label))
    return siamese_features



def convert_to_siamese_features_for_test(test, train):
    siamese_features = []
    test_len = len(test)
    train_len = len(train)
    for idx in range(test_len):
        data_a  = test[idx].input_ids
        data_a_mask = test[idx].input_mask
        data_a_lab = test[idx].label_ids
        for idx_2 in range(train_len):
            data_b = train[idx_2].input_ids
            data_b_mask = train[idx_2].input_mask
            data_b_lab = train[idx_2].label_ids
            label = int(np.where(test[idx].label_ids == train[idx_2].label_ids, 
                             1,0))
            siamese_features.append(SiameseFeatures(feature_id = 'test_{}'.format(int(idx)), 
                                                    data_a = data_a, 
                                                    data_a_mask = data_a_mask, 
                                                    data_a_lab = data_a_lab, 
                                                    data_b = data_b, 
                                                    data_b_mask = data_b_mask, 
                                                    data_b_lab = data_b_lab, 
                                                    label = label))
    return siamese_features