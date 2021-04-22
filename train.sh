yes yes | python3 -m colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 1 --accum 1 \
--triples /home/u9296553/MSMARCO/passage/triples.train.small.tsv \
--root result/msmarco --experiment test --similarity l2 --run msmarco.psg.l2.test
