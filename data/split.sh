#/bin/bash

fpath=("ar.txt"\
       "de.txt"\
       "en.txt"\
       "es.txt"\
       "fa.txt"\
       "fr.txt"\
       "hi.txt"\
       "ja.txt"\
       "ko.txt"\
       "nl.txt"\
       "ru.txt"\
       "ta.txt"\
       "th.txt"\
       "tr.txt"\
       "zh.txt")

for f in "${fpath[@]}"; do
    lines=$(wc -l $f | awk '{print $1}')
    test_lines=$(( $lines/10))
    train_lines=$(( $lines - test_lines))
    echo spliting file $f, total $lines lines, train data $train_lines lines, test data $test_lines lines
    # tail -n$test_lines $f > test_$f
    # head -n$train_lines $f > train_$f

    cat $f | sort -R > temp.txt
    tail -n$test_lines temp.txt > "test_$f"
    head -n$train_lines temp.txt > "train_$f"
    rm temp.txt
done

