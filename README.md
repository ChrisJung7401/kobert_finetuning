# kobert_finetuning

fine-tuning pretrained BERT model with Kobert from SKT

<br>

**Datasets**    
    - NSMC  
    - News data  
    
    
1. Multi-class Classification  
  **Extra implemented**
    - Doc_Stride
    - fp-16 more smoothly
    - multiprocessing on tokenizing
   

<br>

  **How to use**  
  
    python -m torch.distributed.launch --nproc_per_node=4 run_classifier_spm.py \
      --train_file news_tr.txt \
      --eval_file news_te.txt \
      --data_dir /home/advice/notebook/jms/haha/data/ \
      --task_name news \
      --bert_model extract_kobert/kobert_model.bin \
      --model_config extract_kobert/kobert_config.json \
      --num_train_epochs 4 \
      --train_batch_size 16 \
      --eval_batch_size 16 \
      --local_rank 0 \
      --fp16 True \
      --fp16_opt_level O2 \
      --tokenizer /home/advice/notebook/jms/kobert/kobert_news_wiki_ko_cased-1087f8699e.spiece

  
  
1.1 Multi-class Classification with Siamese Network 

1.1.1 Siamese Network with CosineSimilarity

1.1.2 Siamese Network with Average absolute error

<br>

  **How to use**  
  
      python -m torch.distributed.launch --nproc_per_node=4 run_classifier_sms.py \
       --train_file news_tr_no_cls5.txt \
       --eval_file news_te.txt \
       --dev_file news_dev_1.txt \
       --data_dir /home/advice/notebook/jms/haha/data/ \
       --task_name news \
       --bert_model extract_kobert/kobert_model.bin \
       --model_config extract_kobert/kobert_config.json \
       --tokenizer /home/advice/notebook/jms/kobert/kobert_news_wiki_ko_cased-1087f8699e.spiece \
       --train_batch_size 8 \
       --eval_batch_size 32 \
       --num_train_epochs 1 \
       --local_rank 0 \
       --fp16 True \
       --valid_size 3000
   
<br>

2. QA model (working on)  
  - Korquad v1
  
