import string
import regex
from utils.wiki_util import _normalize


puncs = list(string.punctuation)


class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def remove_articles(text):
    return regex.sub(r'\b(a|an|the)\b', ' ', text)


def white_space_fix(text):
    return ' '.join(text.split())


def remove_punc(text):
    exclude = set(puncs)
    return ''.join(ch for ch in text if ch not in exclude)


def lower(text):
    return text.lower()


def normalize_answer(s):
    return white_space_fix(remove_articles(remove_punc(lower(_normalize(s)))))


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def single_ans_em(pred, gold):
    # pred: prediction string
    # gold: a list of gold answer strings
    if type(gold) != list:
        gold = [gold]
    return max(compute_exact(pred, a) for a in gold)


def has_correct_answer(retrieve_doc, answers):
    tokenizer = SimpleTokenizer()
    retrieve_doc = _normalize(retrieve_doc)
    # retrieve_doc = normalize_answer(retrieve_doc)
    retrieve_doc = tokenizer.tokenize(retrieve_doc, uncased=True)

    for single_answer in answers:
        single_answer = _normalize(single_answer)
        # single_answer = normalize_answer(single_answer)
        single_answer = tokenizer.tokenize(single_answer, uncased=True)

        for i in range(0, len(retrieve_doc) - len(single_answer) + 1):
            if single_answer == retrieve_doc[i: i + len(single_answer)]:
                return 1
    return 0
