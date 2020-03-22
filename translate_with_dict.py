import os
import argparse
from collections import defaultdict


class DictTranslator:
    def __init__(self, src, tgt, unk_token='<UNK>'):
        self.tokenizer = get_tokenizer(src)
        self.word_dict = load_bi_dict(f'resources/bi-dict/{src}-{tgt}.txt', unk_token)

    def translate_sent(self, src_sent):
        return ' '.join([self.word_dict[tok] for tok in self.tokenizer(src_sent)])

    def translate_save_corpus(self, src_fpath, out_fpath):
        with open(src_fpath) as src_f, open(out_fpath) as out_f:
            for line in src_f:
                print(self.translate_sent(line), file=out_f)


def get_tokenizer(lang):
    if lang in {'ar', 'ara', 'arabic'}:
        import pyarabic.araby as araby
        return araby.tokenize
    elif lang in {'ja', 'jpn', 'japanese'}: #FIXME: Chan seems to have used different one
        import MeCab
        wakati = MeCab.Tagger("-Owakati")
        return lambda sent: wakati.parse(sent).split()
    elif lang in {'ko', 'kor', 'korean'}:
        from konlpy.tag import Mecab
        m = Mecab()
        return m.morphs
    elif lang in {'zh', 'zho', 'chinese'}:
        import jieba
        return lambda sent: [tok[0] for tok in jieba.tokenize(sent)]
    else:
        return lambda sent: sent.split()

def load_bi_dict(bi_dict_fpath, unk_token):
    bi_dict = defaultdict(lambda: unk_token)
    with open(bi_dict_fpath, 'r') as f:
        for line in f:
            src_word, tgt_word = line.split()
            if src_word not in bi_dict: # use the first entry matched
                bi_dict[src_word] = tgt_word
    return bi_dict

if __name__ == '__main__':

    test_sents = {
        'ar': 'اضربها جيدًا، هيا، تصرف بشكل أفضل، وصل لي كل الشعور.',
        'de': 'jeden Aspekt unserer Art zu lieben und zu leben, zu erziehen und zu führen.',
        'en': 'Hit it good, c\'mon, do better, give me all the feel Oooh',
        'es': 'Muévelo bien, vamos, hazlo mejor, dámelo todo.',
        'fa': 'زلام نیست آن را خیلی بزرگ جلوه دهید اما مخفی کاری نهایتا مخرب خواهد بود.',
        'fr': 'Vas-y à fond, encore plus Fais-moi vibrer Oooh',
        'hi': 'ताकि आप वास्तव में स्थानांतरित कर सकते है स्कूल से लेकर काम तक । और अधिक ।',
        'ja': '壊れた思い出の重なりの下にいる私を。',
        'ko': '그들이 우리를 몰아넣은 혼란 속에서 벗어나.',
        'nl': 'De zaak van mijn vader hield geen stand door zijn ziekte.',
        'ru': 'История гендерной вариативности старше, чем вы думаете.',
        'ta': 'அந்த இடத்தை நான் ஒரு பெண்மனியுடனும், அவரது மகளுடனும் பகிர்ந்து கொள்ள வேண்டியிருந்தது.',
        'tr': 'beni bütün sakinliğinle sarar mısın?',
        'zh': '做得好 来吧不要停 继续 让我感受到你的心情 噢'
    }
    for src, src_sent in test_sents.items():
        tgt = 'en'
        if src == tgt: continue
        translator = DictTranslator(src, tgt)
        res = translator.translate_sent('안녕하세요, 너가 그렇게 번역을 잘 해?')
        print(f'Sample translation from {src} to {tgt}:')
        print(f'\tsrc: {src_sent}')
        print(f'\ttgt: {res}')


