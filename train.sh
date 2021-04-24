yes yes | python3 -m colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 16 --accum 1 \
--triples /home/nlp/GeneralQA/MSMARCO/passage/triples.train.small.tsv \
--resume --resume_optimizer \
--checkpoint /home/nlp/GeneralQA/Col-KBERT/result/msmarco/col_kbert_v2/train.py/msmarco.psg.l2.col_kbert_v2/checkpoints/colbert-300000.dnn \
--root result/msmarco --experiment col_kbert_v2 --similarity l2 --run msmarco.psg.l2.col_kbert_v2
