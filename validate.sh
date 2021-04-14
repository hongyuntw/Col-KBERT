python3 -m colbert.test --amp --doc_maxlen 180 --mask-punctuation --bsize 1024 \
--qrels ~/MSMARCO/passage/qrels.dev.small.tsv \
--topk ~/MSMARCO/passage/top1000.dev  \
--checkpoint /home/u9296553/ColBERT/result/msmarco/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-300000.dnn \
--root result/msmarco --experiment MSMARCO-psg