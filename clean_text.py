import re, string, nltk, contractions
from collections import defaultdict
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from num2words import num2words


class clean_text:

    def __init__(self):
        pass
    
    def rm_whitespace_before_punct(self, text):
        '''
        Removes leading whitespace before punctuation (but not after) for text string.
        :param text: str, text string
        :return: str
        '''
        return re.sub(r'\s+([?,.!\'\"])', r'\1', text)
    

    def remove_URL(self, sentence):
        '''
        Strips URLs if any in text
        :param sentence: string, text
        :return: string, text
        '''
        assert isinstance(sentence, str)
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'', sentence)

    def remove_html(self, sentence):
        '''
        Removes HTLM from text
        :param sentence: str
        :return: str
        '''
        assert isinstance(sentence, str)
        html = re.compile(r'<.*?>')
        return html.sub(r'', sentence)

    def remove_emoji(self, sentence):
        '''
        Removes emoticons, symbols, etc. of text
        :param sentence:str
        :return: str
        '''
        assert isinstance(sentence, str)
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', sentence)

    def remove_contractions(self, sentence):
        '''
        Converts contractions (e.g. he's -> he is) to standard English in a given document.
            Note - does not remove possessive-s after proper noun.
        :param sentence: str, individual document/sentence
        :return: str
        '''
        return contractions.fix(sentence)

    def add_space_if_missing(self, sentence):
        '''
        Adds space after [.,?!] if not followed immediately by space. E.g. end.This -> end. This
        :param sentence: str, individual document/sentence
        :return: str
        '''
        return re.sub(r'(?<=[.,?!])(?=[^\s])', r' ', sentence)

    def remove_punct(self, sentence):
        '''
        Removes punctuation from input string
        :param sentence: str, individual document/sentence
        :return: str
        '''
        assert isinstance(sentence, str)
        table = str.maketrans('', '', string.punctuation)
        return sentence.translate(table)

    def numbers_to_char(self, sentence):
        '''
        Converts numbers and decades (common in dataset) to char representation
        :param sentence: str, individual document/sentence
        :return: str
        '''
        decades = {'20s': 'twenties',
                   "20's": "twenties",
                   '1920s': 'twenties',
                   "1920's": 'twenties',
                   '30s': 'thirties',
                   "30's": "thirties",
                   '1930s': 'thirties',
                   "1930's": 'thirties',
                   '40s': 'forties',
                   "40's": "forties",
                   '1940s': 'forties',
                   "1940's": 'forties',
                   '50s': 'fifties',
                   "50's": "fifties",
                   '1950s': 'fifties',
                   "1950's": 'fifties',
                   '60s': 'sixties',
                   "60's": "sixties",
                   '1960s': 'sixties',
                   "1960's": 'sixties',
                   '70s': 'seventies',
                   "70's": "seventies",
                   '1970s': 'seventies',
                   "1970's": 'seventies',
                   '80s': 'eighties',
                   "80's": "eighties",
                   '1980s': 'eighties',
                   "1980's": 'eighties',
                   '90s': 'nineties',
                   "90's": "nineties",
                   '1990s': 'nineties',
                   "1990's": 'nineties',
                   '2000s': 'two-thousands',
                   "2000's": "two-thousands",
                   '2010s': 'twenty-tens',
                   "2010's": "twenty-tens"}

        expanded_words = []
        for word in sentence.split():
            if word in decades.keys():  # check if decade
                word = decades[word]
            elif word.isnumeric():  # check if numeric
                word = num2words(word)
            else:
                pass  # char string or alphanumeric string
            expanded_words.append(word)
        return ' '.join(expanded_words)

    def remove_stop_words(self, text):
        '''
        Removes stop words (e.g. 'the', 'I') from a Pandas Series
        :param text: pd.Series containing text
        :return: pd.Series
        '''
        assert isinstance(text, pd.core.series.Series)
        stop_words = stopwords.words('english')
        pat = r'\b(?:{})\b'.format('|'.join(stop_words))
        return text.str.replace(pat, '')

    def remove_hash_at(self, sentence):
        '''
        Removes hash ('#') symbol and at symbol ('@') from string
        :param sentence: str
        :return: str
        '''
        assert isinstance(sentence, str)
        return ' '.join(word for word in sentence.split(' ') if not word.startswith(('#', '@')))

    def get_wordnet_pos(self, word):
        '''
        Map POS tag to first character lemmatize() accepts
        :param word: str, word
        :return: list containing a tuple, where first element in tuple is word, second is word type (adjective, verb, etc)
        '''
        assert isinstance(word, str)
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            'J': wordnet.ADJ,
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def lemmatize_string(self, sentence):
        '''
        Lemmatizes string, e.g. converts plural nouns to singular, verbs back to infinitive present case
        :param sentence: string
        :return: list of lemmatized words
        '''
        assert isinstance(sentence, str)
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(w, self.get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)]

    def lematizer(self, text):
        '''
        Runs lematize_sting method across all documents in corpus. Returns pd.Series of lemmatized documents.
        :param text: pd.Series
        :return: pd.Series of lemmatized documents
        '''
        assert isinstance(text, pd.core.series.Series)

        lst = []
        for nr in text.index:
            try:
                lemmatized_words = self.lemmatize_string(text[nr])  # converts string to list
            except TypeError:  # if empty string
                lemmatized_words = ['nan']
            lst.append(lemmatized_words)
        return pd.Series(lst, index=text.index).apply(lambda x: ' '.join(x))

    def enforce_min_word_length(self, text, min_char_word=3):
        '''
        Keeps only words of minimum chararacter length per document
        :param text: pd.Series, input text
        :param min_char_word: int, minimum number of character length of word
        :return: pd.Series with len(word) >= min_char_word for all words in corpus
        '''
        assert isinstance(text, pd.core.series.Series)
        assert isinstance(min_char_word, int)

        temp = [i.split() for i in text]  # converts to list of lists
        modified = [[word for word in sublist if len(word) >= min_char_word] for sublist in temp]
        return pd.Series(modified, index=text.index).apply(lambda x: ' '.join(x))

    def remove_infreq_words(self, text, min_word_freq=1):
        '''
        Removes infrequent words from corpus
        :param text: pd.Series, input text
        :param min_word_freq: int, minimum number of times a word must appear across all documents in corpous to remain
        :return: list of lists
        '''
        assert isinstance(text, pd.core.series.Series)
        assert isinstance(min_word_freq, int)

        temp = [i.split() for i in text]  # converts to list of lists
        freq = defaultdict(int)  # Get freq of each word across all documents
        for indiv_doc in temp:
            for token in indiv_doc:
                freq[token] += 1
        modified = [[token for token in indiv_doc if freq[token] > min_word_freq] for indiv_doc in temp]
        return pd.Series(modified, index=text.index).apply(lambda x: ' '.join(x))

    def run(self, text, no_stop_words=True, remove_punctuation=True, lemmatize=True, min_char_word=None,
            min_word_freq=None):
        '''
        Function calls other functions in class
        :param text: pd.Series of input text
        :param no_stop_words: bool, true if remove stop words from corpus
        :param correct_spelling: bool, true if misspellings should be corrected
        :param lemmatize: bool, whether to lemmatize words or not
        :param min_char_word: Optional - int, minimum characters of word to keep in corpus
        :param min_word_freq: Optional - int, minimum number of times a word must appear in corpus
        :return: pd.Series of corrected text
        '''
        # Basic formatting, removing punctuation, nonstandard characters, whitespace, etc.
        text = text.apply(lambda x: self.remove_URL(x))
        text = text.apply(lambda x: self.remove_html(x))
        text = text.apply(lambda x: self.remove_emoji(x))
        text = text.apply(lambda x: self.remove_hash_at(x))
        text = text.str.lower()  # all lower case
        text = text.apply(lambda x: self.remove_contractions(x))
        text = text.str.lower()  # contractions returns uppercase words, e.g. "I am"
        text = text.apply(lambda x: self.add_space_if_missing(x))
        if remove_punctuation:
            text = text.apply(lambda x: self.remove_punct(x))
        text = text.apply(
            lambda x: re.sub(r"[^A-Za-z0-9^,!?.\/'+]", " ", x))  # keep only regular Latin characters & punctuation
        text = text.apply(lambda x: self.numbers_to_char(x))
        text = text.apply(lambda x: ' '.join(
            s for s in x.split() if not any(c.isdigit() for c in s)))  # remove whole word if alphanumeric
        text = text.fillna(value='')  # any observations with no text, cast to empty string

        if no_stop_words:
            text = self.remove_stop_words(text)

        if lemmatize:  # this can take awhile
            text = self.lematizer(text)

        if min_char_word is not None:
            text = self.enforce_min_word_length(text, min_char_word)

        if min_word_freq is not None:
            text = self.remove_infreq_words(text, min_word_freq)

        # Remove extra whitespace if above introduced any
        text = text.apply(lambda x: re.sub(' +', ' ', x))  # remove extra whitespace between words
        text = text.str.lstrip().str.rstrip()  # strip any whitespace at end or beginning
        text = text.apply(self.rm_whitespace_before_punct) # remove leading whitespace before punctuation

        return text