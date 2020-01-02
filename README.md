# kobert_finetuning

fine-tuning pretrained BERT model with Kobert from SKT

1. Multi-class Classification
  Datasets
    - NSMC
    - News data
  
  How to use
  
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

  
  
1.1 Multi-class Classification with Siamese Network (working on)

2. QA model (working on)
  - Korquad v1
  
