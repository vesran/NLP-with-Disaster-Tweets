from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import re
import pandas as pd
import string


def decontracted(phrase):
    """ From https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
    :param phrase: string
    :return: string
    """
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"ain\'t", "is not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not) | Misses some contractions
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)


def clean(sent):
    """ Removes @user, #, digits, punctuations, double spaces...
    :param sent: string
    :return: clean string
    """
    sent = re.sub('[^%s]' % string.printable, '', sent)
    sent = decontracted(sent)
    sent = " ".join(text_processor.pre_process_doc(sent.lower()))
    sent = re.sub('<.*?>', '', sent)
    sent = re.sub(r'\w*\d\w*', '', sent)
    sent = re.sub(f"[{string.punctuation}]", " ", sent)
    sent = re.sub(' +', ' ', sent)
    return sent


# Import CSV & drop duplicates !
train = pd.read_csv('./data/train.csv').drop_duplicates(keep='first')
test = pd.read_csv('./data/test.csv')
ground_truth = pd.read_csv('./data/ground_truth.csv')
test['target'] = ground_truth['target']
train['source'], test['source'] = 'train', 'test'
train = train.drop(['location'], axis=1)  # Dropping location : useless column
test = test.drop(['location'], axis=1)

# Clean tweets
train['clean_text'] = train['text'].apply(clean)
test['clean_text'] = test['text'].apply(clean)


# Handle ain't
def correct_be(sent):
    """ 'Ain't' contraction might be misreplaced by 'is not' all the time.
    :param sent: string
    :return: string
    """
    sent = re.sub('i is', 'i am', sent)
    sent = re.sub('you is', 'you are', sent)
    sent = re.sub('we is', 'we are', sent)
    sent = re.sub('they is', 'they are', sent)
    return sent


train['clean_text'] = train['clean_text'].apply(correct_be)
test['clean_text'] = test['clean_text'].apply(correct_be)

# Remove duplicates caused by cleaning
train = train.drop_duplicates(keep='first')
print(f'train.shape=({train.shape})')


# Save CSV
df = pd.concat([train, test])
outfile = "./data/clean_entire_corpus.csv"
df.to_csv(outfile, index=False)


