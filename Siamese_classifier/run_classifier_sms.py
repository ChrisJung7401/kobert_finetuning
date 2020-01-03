
"""
Run koBERT classifier with siamese network

MADE BY MINSUNG JUNG

INSPIRED BY HUGGING FACE transformers & FAST-BERT 


v.0.0


"""

import csv
import pickle
import argparse
import logging
import os
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
from apex import amp

import torch
from datetime import date
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
from tensorboardX import SummaryWriter

from tokenization_spm import *
from data_cls_sms import *
from func_cls_sms import *

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def parse():
    parser = argparse.ArgumentParser( description = "Pytorch kobert classifier")
    parser.add_argument("--fund_dir", default='/home/advice/notebook/jms/우리은행',type=Path)
    parser.add_argument("--train_file", default=None,type=Path,required=True,)
    parser.add_argument("--eval_file", default=None,type=Path,required=True,)
    parser.add_argument("--data_dir", default=None,type=Path,required=True,)
    parser.add_argument("--task_name", default=None,type=str,required=True,)
    parser.add_argument("--no_cuda", default=False,type=bool,)
    parser.add_argument("--bert_model", default=None,type=Path,required=True,)
    parser.add_argument("--model_config", default=None,type=Path,required=True,)
    parser.add_argument("--output_dir", default=None,type=Path,)
    parser.add_argument("--tokenizer", default=None,type=Path,required=True,)
    parser.add_argument("--max_seq_length", default=512,type=int,)
#     parser.add_argument("--doc_stride", default=128,type=int,
#                        help="if you donot want doc_stride, 999")
    parser.add_argument("--do_train", default=True,type=bool,)
    parser.add_argument("--do_eval", default=True,type=bool,)
    parser.add_argument("--train_batch_size", default=16,type=int,)
    parser.add_argument("--eval_batch_size", default=16,type=int,)
    parser.add_argument("--learning_rate", default=3e-5,type=float,)
    parser.add_argument("--num_train_epochs", default=4,type=int,)
    parser.add_argument("--warmup_proportion", default=0.1,type=float,)
    parser.add_argument("--local_rank", default=-1,type=int,)
    parser.add_argument("--seed", default=42,type=int,)
    parser.add_argument("--gradient_accumulation_steps", default=1,type=int,)
    parser.add_argument("--optimize_on_cpu", default=False,type=bool,)
    parser.add_argument("--fp16", default=False,type=bool,)
    parser.add_argument("--fp16_opt_level", default='O2',type=str,)
    parser.add_argument("--loss_scale", default=0,type=int,)
    parser.add_argument("--logging_steps", default=100,type=int,)
    parser.add_argument("--max_grad_norm", default=1.0,type=float,)
    parser.add_argument("--num_workers", default=8,type=int,)
    args = parser.parse_args()
    return args
    
    
def main():
    args = parse()
    if args.local_rank!=-1:
        current_env = os.environ.copy()
        args.local_rank=int(current_env["LOCAL_RANK"])
    
    
    print(args)
    ## setting seeds
    set_seeds(args.seed)
    ## set directories
    if args.output_dir ==None:
        today = str(date.today()).replace('-', '')
        output_dir = args.fund_dir/'output_dir'/today
        output_dir.mkdir(exist_ok = 'True')
    else:
        output_dir = args.output_dir
        output_dir.mkdir(exist_ok = 'True')
    args.model_config = args.fund_dir /args.model_config 
    args.bert_model = args.fund_dir / args.bert_model
    args.data_path = args.fund_dir/args.data_dir
    
    processors = {args.task_name: MultiClassTextProcessor}
    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    if args.task_name =='news':
        lab = ['0','1','2','3','4','5']
    elif args.task_name =='nsmc':
        lab = ['0','1']
    ## set processor
    processor = processors[task_name](data_path = args.data_dir, 
                                      train_file = args.train_file, 
                                      test_file = args.eval_file, 
                                      labels = lab,
                                     )
    label_list = processor.get_labels()
    num_labels = len(label_list)
    ## get tokenizer
    tokenizer = BERTSPMTokenizer.from_pretrained(args.tokenizer)
    train_examples = None
    num_train_steps = None
    train_feature_name = str(args.train_file)[:-4]+'_'+str(args.max_seq_length)
    cached_train_features_file = args.data_path/train_feature_name
    if args.do_train:
        train_examples = processor.get_train_examples()
        try:
            with open(cached_train_features_file, 'rb') as reader:
                train_features = pickle.load(reader)
        except:
            
            train_features = convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, tokenizer)
            logger.info("  Saving train features into cached file %s", cached_train_features_file)
            with open(cached_train_features_file, "wb") as writer:
                pickle.dump(train_features, writer)
        num_train_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    ##### Setup GPU parameters######
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16))
    
    if args.local_rank not in [-1, 0]:
        # 모델을 한gpu에 올리겟다... 인데 다올라감... 수정 및 확인 필요
        torch.distributed.barrier()
        
        
    bert_config = modeling.BertConfig.from_json_file(args.model_config)
    model = BertForSiameseClassification(bert_config, num_labels = num_labels)
    model.bert.load_state_dict(torch.load(args.bert_model))
    if args.local_rank == 0:
        # 모델을 한gpu에 올리겟다... 인데 다올라감... 수정 및 확인 필요
        torch.distributed.barrier()
    model.to(device)
    
        ## got info from transformers that we have to check if fp16 is set 
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    logger.info("***Now, Model is on the device!!!***")
    
    
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.contrib.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False)
        logger.info("***Now, Model &optimizer setting is finished***")

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)
        logger.info("***Now, Model &optimizer setting is finished2***")

    scheduler = CyclicLR(optimizer, base_lr=2e-5, max_lr=5e-5, step_size=2500, last_batch_iteration=0)

    if args.fp16:
        if args.loss_scale == 0:
            model, optimizer = amp.initialize(model, optimizer, 
                                          opt_level=args.fp16_opt_level, 
                                          keep_batchnorm_fp32=False, 
                                          loss_scale = "dynamic")
        else:
            model, optimizer = amp.initialize(model, optimizer, 
                                              opt_level=args.fp16_opt_level, 
                                              keep_batchnorm_fp32=False, 
                                              loss_scale = args.loss_scale)
        logger.info("***Now, Model is amp initialized!!!***")
        
    if args.local_rank != -1:
#         model = torch.nn.parallel.DistributedDataParallel(
#             model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
#         )
#         logger.info("***Now, Model parallelized!!!***")
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model, delay_allreduce=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        logger.info("***Now, Model parallelized!!!***")
    
    
    

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Num features = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        siamese_features = convert_to_siamese_features(train_features)
        all_data_a = torch.tensor([f.data_a for f in siamese_features],dtype=torch.long)
        all_data_a_mask = torch.tensor([f.data_a_mask for f in siamese_features],dtype=torch.float)
        all_data_b = torch.tensor([f.data_b for f in siamese_features],dtype=torch.long)
        all_data_b_mask = torch.tensor([f.data_b_mask for f in siamese_features],dtype=torch.float)
        all_label = torch.tensor([f.label for f in siamese_features])
        train_data = TensorDataset(all_data_a, all_data_a_mask, all_data_b, all_data_b_mask, all_label)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, 
                                      pin_memory=True, batch_size=args.train_batch_size, shuffle = False, 
                                      num_workers = args.num_workers)
        
        tensorboard_dir = output_dir / "tensorboard"
        tensorboard_dir.mkdir(exist_ok=True)
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter(tensorboard_dir)

        global_step = 0
        tr_loss, logging_loss, epoch_loss = 0.0, 0.0, 0.0
        model.train()
        for i_ in tqdm(range(int(args.num_train_epochs)), desc="Epoch"):

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):

                batch = tuple(t.to(device) for t in batch)
                dat_a, mask_a, dat_b, mask_b, lab= batch
                logit, _ = model(dat_a, mask_a, dat_b, mask_b)
                loss_func = torch.nn.CrossEntropyLoss() 
                loss = loss_func(logit, lab)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                tr_loss += loss.item()
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
        #             scheduler.batch_step()
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    model.zero_grad()#optimizer.zero_grad()
                    global_step += 1

                if args.logging_steps>0 and global_step % args.logging_steps==0 and args.local_rank in [-1, 0]:
                    tb_writer.add_scalar("loss",(tr_loss - logging_loss) / args.logging_steps,global_step,)
#                     tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                logging_loss = tr_loss
            logger.info('Loss after epoc {}'.format(tr_loss / nb_tr_steps))
            logger.info('Eval after epoc {}'.format(i_+1))
        
        if args.local_rank in [-1, 0]:
            tb_writer.close()
    
    
    # Save a trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    model_name = "finetuned_{}__sms_classifier_pytorch_model_m{}_e{}.bin".format(args.task_name, args.max_seq_length, args.num_train_epochs)
    output_model_file = output_dir/model_name
    torch.save(model_to_save.state_dict(), output_model_file)
    
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        
        
        test_examples = processor.get_test_examples()
        test_features = convert_examples_to_features(
        test_examples[:1000], label_list, args.max_seq_length, tokenizer)
#         train_features_picked = 
        test_set = convert_to_siamese_features_for_test(test_features, train_features)
        all_test_data = torch.tensor([f.data_a for f in test_set],dtype=torch.long)
        all_test_data_mask = torch.tensor([f.data_a_mask for f in test_set],dtype=torch.float)
        all_compare_data = torch.tensor([f.data_b for f in test_set],dtype=torch.long)
        all_compare_data_mask = torch.tensor([f.data_b_mask for f in test_set],dtype=torch.float)
        test_ans = [f.data_a_lab[0] for f in test_set]
        test_id = [f.feature_id for f in test_set]
        train_lab = [f.data_b_lab[0] for f in test_set]
        test_data = TensorDataset(all_test_data, all_test_data_mask, all_compare_data, all_compare_data_mask)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, 
                                     pin_memory=True, batch_size=args.eval_batch_size, shuffle = False, 
                                      num_workers = args.num_workers)
        model.eval()
        all_logits = None
        for step, batch in enumerate(tqdm(test_dataloader, desc="Prediction Iteration")):
            batch = tuple(t.to(device) for t in batch)
            test_data, test_data_mask, compare_data, compare_data_mask = batch
            with torch.no_grad():
                _,logits = model(test_data, test_data_mask, compare_data, compare_data_mask)
                if all_logits is None:
                    all_logits = logits.detach().cpu().numpy()
                else:
                    all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)


        result = pd.DataFrame({'test_id': test_id, 'train_label':train_lab, 'prob':all_logits[:,1],'real':test_ans})
        
        
        result_pivot = pd.pivot_table(result, values='prob', 
                              index = ['test_id', 'real'], 
                              columns = ['train_label'], aggfunc = np.mean)
    
        result_pivot.loc[:,'pred'] = result_pivot.apply(lambda row: np.argmax(row), axis = 1)
        result_pivot = result_pivot.reset_index()
        result_pivot[result_pivot['real'] == result_pivot['pred']].shape[0]/result_pivot.shape[0]
        result_pivot = result_pivot.reset_index()
        acc = result_pivot[result_pivot['real'] == result_pivot['pred']].shape[0]/result_pivot.shape[0]
        
        logger.info('Accuracy of the model on test set: {}'.format(acc))
    
    
if __name__ == "__main__":
    main()    
    
    