yes yes | python3 -m colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 80 --accum 1 \
--triples /home/u9296553/MSMARCO/passage/triples.train.small.tsv \
--root result/msmarco --experiment col_kbert_v1 --similarity l2 --run msmarco.psg.l2.col_kbert_v1
