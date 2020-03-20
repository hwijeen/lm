langs=(ar de en es fa fr hi ja ko nl ru ta)

for l in ${langs[@]};do
        for n in {2..2}; do
            python ngram_lm.py --lang $l --n $n
            python ngram_lm.py --lang $l --n $n --char
        done
done
