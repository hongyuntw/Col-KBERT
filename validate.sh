python3 -m colbert.test --amp --doc_maxlen 180 --mask-punctuation --bsize 1024 \
--topk ~/MSMARCO/passage/top1000.eval  \
--checkpoint /home/u9296553/ColBERT/result/msmarco/mask_v2/train.py/msmarco.psg.l2.mask_v2/checkpoints/colbert-300000.dnn \
--root result/msmarco --experiment mask_v2