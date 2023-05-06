import re
import cn2an
import jieba
from pypinyin import lazy_pinyin, BOPOMOFO

# chinese_cleaners
_pad        = '_'
_punctuation = '，。！？—…'
_letters = 'ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧㄨㄩˉˊˇˋ˙ '

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters)
symbols_joined = '_，。！？—…ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧㄨㄩˉˊˇˋ˙ '

# create a dictionary to map each character to its index
symbols_to_index = {char: index for index, char in enumerate(symbols)}

# remove unexpected symbols from text
symbols_filter = re.compile(f"[^{symbols_joined}]")

# redundant symbols filter
red_symbols_filter = re.compile('[' + _punctuation + ']{2,}')

# Special symbol ids
SPACE_ID = symbols.index(" ")

# replace emoji symbols and full stop symbols
period_pattern = re.compile(
    # Match Unicode block for emoticons and miscellaneous symbols
    '['
    '\U0001F300-\U0001F5FF'
    # Match Unicode block for additional emoticons
    '\U0001F900-\U0001F9FF'
    # Match Unicode block for emojis
    '\U0001F600-\U0001F64F'
    # Match Unicode block for emojis, symbols, and pictographs
    '\U0001F680-\U0001F6FF'
    # Match Unicode block for additional emojis and symbols
    '\U0001F910-\U0001F96C'
    # Match Unicode block for flags (iOS)
    '\U0001F1E0-\U0001F1FF'
    # Match Unicode block for flags
    '\U0001F100-\U0001F1FF'
    # Match Unicode block for keycaps, and other symbols
    '\U00002702-\U000027B0'
    # Match Unicode block for regional indicator symbols
    '\U0001F1E6-\U0001F1FF'
    # Match Unicode block for dingbats
    '\U00002700-\U000027BF'
    # Match full stop symbols
    '；“”()<>《》「」【】'
    ']',
    flags=re.UNICODE
)

# replace half stop symbols
comma_pattern = re.compile("[、：]")

# chinese char
chin_pattern = re.compile('[\u4e00-\u9fff]')

# single phonetics
single_pattern = re.compile(r'([\u3105-\u3129])$')

# no end
no_end_pattern = re.compile(r'([ˉˊˇˋ˙])$')

# List of (Latin alphabet, bopomofo) pairs:
_latin_to_bopomofo = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
    ('a', 'ㄟˉ'),
    ('b', 'ㄅㄧˋ'),
    ('c', 'ㄙㄧˉ'),
    ('d', 'ㄉㄧˋ'),
    ('e', 'ㄧˋ'),
    ('f', 'ㄝˊㄈㄨˋ'),
    ('g', 'ㄐㄧˋ'),
    ('h', 'ㄝˇㄑㄩˋ'),
    ('i', 'ㄞˋ'),
    ('j', 'ㄐㄟˋ'),
    ('k', 'ㄎㄟˋ'),
    ('l', 'ㄝˊㄛˋ'),
    ('m', 'ㄝˊㄇㄨˋ'),
    ('n', 'ㄣˉ'),
    ('o', 'ㄡˉ'),
    ('p', 'ㄆㄧˉ'),
    ('q', 'ㄎㄧㄡˉ'),
    ('r', 'ㄚˋ'),
    ('s', 'ㄝˊㄙˋ'),
    ('t', 'ㄊㄧˋ'),
    ('u', 'ㄧㄡˉ'),
    ('v', 'ㄨㄧˉ'),
    ('w', 'ㄉㄚˋㄅㄨˋㄌㄧㄡˋ'),
    ('x', 'ㄝˉㄎㄨˋㄙˋ'),
    ('y', 'ㄨㄞˋ'),
    ('z', 'ㄗㄟˋ')
]]

def number_to_chinese(text):
    numbers = re.findall(r'\d+(?:\.?\d+)?', text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    return text

def chinese_to_bopomofo(text):
    # replace all emoji symbols with a comma
    text = period_pattern.sub("。", text)
    text = comma_pattern.sub("，", text)
    words = jieba.lcut(text, cut_all=False)
    text = ''
    for word in words:
        bopomofos = lazy_pinyin(word, BOPOMOFO)
        if not chin_pattern.search(word):
            text += word
            continue
        for i in range(len(bopomofos)):
            bopomofos[i] = single_pattern.sub(r'\1ˉ', bopomofos[i])
        if text != '':
            text += ' '
        text += ''.join(bopomofos)
    return text

def latin_to_bopomofo(text):
    for regex, replacement in _latin_to_bopomofo:
        text = regex.sub(replacement, text)
    return text

def chinese_cleaners(text):
    '''Pipeline for Chinese text'''
    text = number_to_chinese(text)
    text = chinese_to_bopomofo(text)
    text = latin_to_bopomofo(text)
    text = no_end_pattern.sub(r'\1。', text)
    text = symbols_filter.sub("", text)
    text = red_symbols_filter.sub("。", text)
    return text

def clean_text_to_interspersed_sequence(clean_text):
    sequence = [0] * (len(clean_text) * 2 + 1)
    sequence[1::2] = [symbols_to_index[char] for char in clean_text]
    return sequence

def text_to_sequence(text, max_length=20):
    clean_text = chinese_cleaners(text)
    """if len(clean_text) <= max_length:
        return [clean_text_to_interspersed_sequence(clean_text)]
    # Split text into sentences using regex pattern for Chinese punctuation
    sentences = re.split('[。？！]', clean_text)
    return [clean_text_to_interspersed_sequence(sentence) for sentence in sentences if sentence]
    """
    return [clean_text_to_interspersed_sequence(clean_text)]