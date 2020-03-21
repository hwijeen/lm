# --- Interpolated bi-gram model code example
# by Graham Neubig

import sys
import math
import argparse
from collections import defaultdict

p = argparse.ArgumentParser(description="""Code to train an extremely simple linearlly interpolated bigram language model""")
p.add_argument("--lang", default='ar', help="Language to train on")
p.add_argument("--n", type=int, default=2, help="n of ngram")
p.add_argument("--char", action='store_true', help="Character level ngram")
p.add_argument("--print_probs", action="store_true", help="Whether to print probabilities for each word")
p.add_argument("--skip_unk", action="store_true", help="Skip unknown words in calculating probabilities")
p.add_argument("--vocab_size", type=int, help="Total size of the vocabulary", default=1e7)
p.add_argument("--unk_alpha", type=float, help="The amount of probability to assign to unknown words", default=0.01)
p.add_argument("--uni_alpha", type=float, help="The amount of probability to assign to unigrams", default=0.20)
args = p.parse_args()
args.train_file = f'data/train_{args.lang}.txt'
args.test_file = f'data/train_{args.lang}.txt'
# args.test_file = f'data/test_{args.lang}.txt'
datasize = len(open(args.train_file).readlines())
if args.char:
    print(f"{args.n}-gram for {args.lang}, char level, data size {datasize} sentences")
else:
    print(f"{args.n}-gram for {args.lang}, word level, data size {datasize} sentences")

N = args.n
VOCAB_SIZE = args.vocab_size
UNK_ALPHA = args.unk_alpha
UNI_ALPHA = args.uni_alpha

# If we're skipping unknown words, set the interpolation probability of unknowns to zero
if args.skip_unk:
    BI_ALPHA = 1.0 - UNI_ALPHA
    UNK_ALPHA = 0.0
else:
    assert(UNK_ALPHA >= 0 and UNK_ALPHA <= 1)
    assert(UNI_ALPHA >= 0 and UNI_ALPHA <= 1)
    assert(UNK_ALPHA + UNI_ALPHA <= 1)
    BI_ALPHA = 1.0 - UNK_ALPHA - UNI_ALPHA

# tokenization
import ast
if args.char:
    # split_f = lambda line: [char for char in line.strip()]
    split_f = lambda line: [char for char in ' '.join(ast.literal_eval(line.strip()))]
else:
    # split_f = lambda line: [w.lower() for w in line.strip().split(" ")]
    # split_f = lambda line: line.strip().split(" ")
    split_f = lambda line: ast.literal_eval(line.strip())



# Read in the training data
train_counts = defaultdict(lambda: 0)
train_ctxts = defaultdict(lambda: 0)
with open(args.train_file, "r") as f:
    for line in f:
        sent = split_f(line) + ["<s>"]
        # sent = line.strip().split(" ") + ["<s>"]
        ngram = ["<s>"] * N
        for word in sent:
            ctxt = ngram[1:]
            ngram = ctxt + [word]
            for i in range(N):
                train_ctxts[tuple(ctxt[i:])] += 1
                train_counts[tuple(ngram[i:])] += 1

# for k, v in train_counts.items():
#     if k[0] == 'pittsburgh':
#         print(k, v)
# sys.exit(0)

# Calculate on test
alpha = [UNK_ALPHA, UNI_ALPHA, BI_ALPHA]
lls = 0
words = 0

with open(args.test_file, "r") as f:
    for line in f:
        sent = split_f(line) + ["<s>"]
        # sent = line.strip().split(" ") + ["<s>"]
        ngram = ["<s>"] * N
        for word in sent:
            ctxt = ngram[1:]
            ngram = ctxt + [word]
            all_probs = [1.0 / VOCAB_SIZE]
            for i in range(N)[::-1]:
                if tuple(ngram[i:]) in train_counts:
                    all_probs.append(train_counts[tuple(ngram[i:])] / train_ctxts[tuple(ctxt[i:])])
                else:
                    all_probs.append(0.0)
            total = 0.0
            for prob, alph in zip(all_probs, alpha):
                total += prob * alph
            if args.print_probs:
                print(' '.join([str(x) for x in [word]+all_probs+[total]]))
            if not (args.skip_unk and total == 0.0):
                lls += math.log(total)
        if args.print_probs:
             print()
        words += len(sent)-1

# Print out the results
my_score = math.exp(-lls/words)
print ("perplexity at alpha=%r: %f\n" % (alpha, my_score), file=sys.stderr)
