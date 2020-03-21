# langs=(ar de en es fa fr hi ja ko nl ru ta tr zh)
langs=(tr)

for l in ${langs[@]};do
        python ngram_lm.py --lang $l --n 2 --vocab_size 20000 --uni_alpha 0.0
        # for n in {2..2}; do
        #     python ngram_lm.py --lang $l --n $n
        #     python ngram_lm.py --lang $l --n $n --char
        # done
done
