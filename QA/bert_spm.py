from __future__ import absolute_import, division, print_function

import collections
import logging
import math

import numpy as np
import torch
import sys

sys.path.append('/home/advice/notebook/jms/우리은행//pytorch-pretrained-BERT/examples/')
sys.path.append('/home/advice/notebook/jms/우리은행//pytorch-pretrained-BERT/')

from run_squad_spm import *
from  pytorch_pretrained_bert import modeling
import torch

from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from utils import (get_answer, input_to_squad_example,
                   squad_examples_to_features, to_list)

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])



class QA:

    def __init__(self,model_path, tokenizer_path, config_path, cpu=False):
        self.max_seq_length = 384
        self.doc_stride = 128
        self.do_lower_case = True
        self.max_query_length = 64
        self.n_best_size = 20
        self.max_answer_length = 30
        self.tokenizer = BERTSPMTokenizer.from_pretrained(tokenizer_path)
        self.config = modeling.BertConfig.from_json_file(config_path)
        self.model = modeling.BertForQuestionAnswering(self.config)
        self.model.load_state_dict(torch.load(model_path))
        if (torch.cuda.is_available()) & (cpu ==False):
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model.to(self.device)
        self.model.eval()

    
    def predict(self,passage :str,question :str):
        example = input_to_squad_example(passage,question)
        features = squad_examples_to_features(example,self.tokenizer,self.max_seq_length,self.doc_stride,self.max_query_length)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=1)
        all_results = []
        for batch in eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2]  
                        }
                example_indices = batch[3]
                outputs = self.model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                result = RawResult(unique_id    = unique_id,
                                    start_logits = to_list(outputs[0][i]),
                                    end_logits   = to_list(outputs[1][i]))
                all_results.append(result)
        answer = get_answer(example,features,all_results,self.n_best_size,self.max_answer_length,self.do_lower_case)
        return answer
