import argparse
import ast

from nltk.util import ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import NgramCounter, Vocabulary, KneserNeyInterpolated, Laplace
from tqdm import tqdm


p = argparse.ArgumentParser(description="""Code to train an extremely simple linearly interpolated bigram language model""")
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
args.test_file = f'data/test_100_{args.lang}.txt'

datasize = len(open(args.train_file, encoding='utf-8').readlines())

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
if args.char:
    split_f = lambda line: list(' '.join(ast.literal_eval(line.strip())))
else:
    split_f = lambda line: ast.literal_eval(line.strip())


def compute_ppl(model, tokens, n):
    sent_ngrams = ngrams(tokens, n,
                         pad_left=True, pad_right=True,
                         left_pad_symbol='<s>',
                         right_pad_symbol='</s>')
    return model.perplexity(sent_ngrams)

# train = []
# vocab = Vocabulary()
with open(args.train_file, "r", encoding='utf-8') as f:
    lines = [split_f(l) for l in f.readlines()]
    #    for line in f:
#        tokens = split_f(line)

train, vocab = padded_everygram_pipeline(N, lines)
# LM = KneserNeyInterpolated(N)
LM = Laplace(N)
LM.fit(train, vocab)

print("Finished LM training")

# ppl_train = [compute_ppl(LM, l, N) for l in lines]
# print(f"Training set perplexity : {sum(ppl_train) / len(ppl_train)}")

with open(args.test_file, "r", encoding='utf-8') as f:
    test_lines = [split_f(l) for l in f.readlines()]

ppl_test = [compute_ppl(LM, l, N) for l in test_lines]
print(f"Test set perplexity : {sum(ppl_test) / len(ppl_test)}")