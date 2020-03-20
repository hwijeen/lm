#/bin/bash

fpath=("ar_sents.txt"\
       "de_sents.txt"\
       "en_sents.txt"\
       "es_sents.txt"\
       "fa_sents.txt"\
       "fr_sents.txt"\
       "hi_sents.txt"\
       "ja_sents.txt"\
       "ko_sents.txt"\
       "nl_sents.txt"\
       "ru_sents.txt"\
       "ta_sents.txt"\
       "th_sents.txt"\
       "tr_sents.txt"\
       "zh_sents.txt")

for f in "${fpath[@]}"; do
    lines=$(wc -l $f | awk '{print $1}')
    test_lines=$(( $lines/10))
    train_lines=$(( $lines - test_lines))
    echo spliting file $f, total $lines lines, train data $train_lines lines, test data $test_lines lines
    # tail -n$test_lines $f > test_$f
    # head -n$train_lines $f > train_$f

    cat $f | sort -R > temp.txt
    tail -n$test_lines temp.txt > test_$f
    head -n$train_lines temp.txt > train_$f
    rm temp.txt
done

