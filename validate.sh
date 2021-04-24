CUDA_VISIBLE_DEVICES=0 python3 -m colbert.test --amp --doc_maxlen 180 --mask-punctuation --bsize 1024 \
--topk /home/nlp/GeneralQA/MSMARCO/passage/top1000.dev  \
--qrels /home/nlp/GeneralQA/MSMARCO/passage/qrels.dev.small.tsv \
--checkpoint /home/nlp/GeneralQA/Col-KBERT/result/msmarco/col_kbert_v2/train.py/msmarco.psg.l2.col_kbert_v2/checkpoints/colbert-300000.dnn \
--root result/msmarco --experiment col_kbert_v2