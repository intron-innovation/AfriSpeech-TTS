import re

def clean_text(text):
    """
    post processing to normalized reference and predicted transcripts
    :param text: str
    :return: str
    """
    # remove multiple spaces
    text = re.sub(r"\s\s+", " ", text)
    # Define a dictionary for replacements
    replace_dict = {
    '>': '',
    '\t': ' ',
    '\n': '',
    ' comma,': ',',
    ' koma,': ' ',
    ' coma,': ' ',
    ' full stop.': '.',
    ' full stop': '.',
    ',.': '.',
    ',,': ',',
    '%': ' percent',
    '&': 'and',
    '1': ' one ',
    '2': ' two ',
    '3': ' three ',
    '4': ' four ',
    '5': ' five ',
    '6': ' six ',
    '7': ' seven ',
    '8': ' eight ',
    '9': ' nine ',
    '*': ' asterisk ',
    '+': ' plus ',
    '0': ' zero ',
    '\xa0': ' ',
    '\u2008': ' ',
    '–': ' ',
    '’': ' ',
    '“': ' ',
    '”': ' ',
    'Mr': 'Mister',
    'Mrs': 'Mistress',
    'Ms': 'Miss',
    'Miss': 'Miss',
    'Alh': 'Alhaji',
    'Prof': 'Professor',
    'Dr': 'Doctor',
    'Engr': 'Engineer',
    'Sir': 'Sir',
    'Hon': 'Honourable',
    'Gen': 'General',
    ':': ' colon ',
    ';': ' semicolon ',
    '(': ' open bracket ',
    ')': ' close bracket ',
    '  ': ' '
}


    # Perform replacements
    for old, new in replace_dict.items():
        text = text.replace(old, new)

    # Strip trailing spaces and convert to lowercase
    text = text.strip()

    text = " ".join(text.split())
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\-\?\:\'\/\(\)\[\]\+\%]", '', text)
    return text


