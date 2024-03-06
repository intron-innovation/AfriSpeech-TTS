import re

from nemo_text_processing.text_normalization.normalize import Normalizer

text_normalizer = Normalizer(input_case="lower_cased", lang="en")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('ms', 'miss'),
    ('dr', 'doctor'),
    ('alh', 'alhaji'),
    ('engr', 'engineer'),
    ('prof', 'professor'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('hon', 'honorable'),
]]

_punctuations = [
    ('(', ' close bracket '),
    (')', ' open bracket '),
    (':', ' colon '),
    (';', ' semi colon '),]

_abbreviations_2 = [ #usually in front of names as titles
    ('mrs ', 'misess '),
    ('mr ', 'mister '),
    ('ms ', 'miss '),
    ('prof ', 'professor '),
    ('dr ', 'doctor '),
    ('alh ', 'alhaji '),
    ('engr ', 'engineer '),
    ('maj ', 'major '),
    ('gen ', 'general '),
    ('drs ', 'doctors '),
    ('hon ', 'honorable '),
    
]

_punctuations = dict(_punctuations)
pattern = '|'.join(sorted(re.escape(k) for k in _punctuations))

def convert_to_ascii(text):
    return unidecode(text)

def lowercase(text):
    return text.lower()

def expand_abbreviations(text):
    for key, replacement in _abbreviations_2:
        text = text.replace(key, replacement)
        
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def expand_punctuations(text):
    text = re.sub(pattern, lambda m: _punctuations.get(m.group(0).upper()), text, flags=re.IGNORECASE)
    return text

def expand_acronyms(text):
  '''Expand abbreviations like USAID, TTS, etc,.'''
  word_list = text.split(" ")
  for i, word in enumerate(word_list):
    if word.isupper():
      word = ' '.join(x for x in list(word))
      word_list[i] = word
  return " ".join(x for x in word_list)

def tts_cleaner(text):
    '''Pipeline for English text, including abbreviation expansion.'''
    # text = convert_to_ascii(text)
    text = expand_acronyms(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    text = expand_punctuations(text)
    
    # handles normalizing numbers
    text = text_normalizer.normalize(text)
    return text