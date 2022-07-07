import html
from xml.sax.saxutils import unescape as xmlunescape
import re
import unicodedata
import json


# get unicode info by unicode point
def get_unicode_info(unicode_point, is_code=True):
    in_s = ord(unicode_point)
    if is_code is True:
        out_s = chr(in_s)
    else:
        out_s = unicode_point
    return unicodedata.category(out_s), unicodedata.name(out_s)


# get unicode info from a string
def get_unicode_info_from_string(line):
    codes = []
    for s in line:
        try:
            codes.append((s, unicodedata.category(s), unicodedata.name(s)))
        except ValueError:
            codes.append((s, unicodedata.category(s), 'no named'))
    return codes


# get unicode name by character
def get_unicode_name(ch):
    try:
        return unicodedata.name(ch)
    except ValueError:
        return None


# unescaped xml, html
def unescaped(s):
    # unescaped xml
    s = xmlunescape(s)
    # unescaped html
    return html.unescape(s)


# remove xml and html tag
def remove_tags(s):
    return re.sub('<.*?>', ' ', s)



unicode_names_regex = {'ko': 'HANGUL SYLLABLE', 'ja': 'HIRAGANA|KATAKANA|CJK', 'zh': 'CJK', 'en': 'LATIN', 'de': 'LATIN'}


def ratio_unicode(lang, text):
    if len(text) == 0:
        return 0.0
    temp_text = str(text).replace(' ', '')
    regex_str = unicode_names_regex[lang]
    # validate regex
    if regex_str is None:
        print('Cannot find language matched.')
        exit(1)

    try:
        matched = len([ch for ch in temp_text if re.match(regex_str, get_unicode_name(ch))])
    except TypeError:
        print('error in string: {0}'.format(temp_text))
        return 0.0
    return round(matched / len(temp_text), 2)


# Transform full width character to half width character
def full_to_half(text):
    return unicodedata.normalize('NFKC', text)



