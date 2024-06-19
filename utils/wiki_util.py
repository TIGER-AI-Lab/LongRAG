import unicodedata
import html
import re
from urllib.parse import unquote


def _normalize(text):
    """
    Normalize Unicode strings. Necessary for text which contains non-ASCII characters.
    """
    return unicodedata.normalize("NFD", text)


def convert_html(text):
    """
    Convert HTML entities entity back to character.
    """
    return html.unescape(text)


def get_hyperlink(text, abstract=None):
    """
    Get abstract of each page and hyperplinks within the whole page.
    :param text: wikipedia text, with hyperlinks
    """
    if abstract is None:
        parts = text.split("\n\n", 1)
        if len(parts) > 1:
            abstract = parts[1].split('\n')[0]
        else:
            abstract = ""

    abs_hyperlink = re.findall(r'<a href="([^"]+)">', abstract)
    full_hyperlink = re.findall(r'<a href="([^"]+)">', text)
    abs_hyperlink = [_normalize(unquote(link)) for link in abs_hyperlink]
    full_hyperlink = [_normalize(unquote(link)) for link in full_hyperlink]
    return abs_hyperlink, full_hyperlink


def remove_hyperlink(text, abstract=False):
    """
    Remove the hyperlink from the text.
    :param text: wikipedia text, with hyperlinks
    """
    if abstract:
        text = text.split("\n\n", 1)
        if len(text) > 1:
            text = text[1].split('\n')[0]
        else:
            text = ""
    text = _normalize(text)
    clean_text = re.sub(r'<a href="[^"]+">([^<]+)</a>', r'\1', text)
    clean_text = re.sub(r'\n\n?', ' ', clean_text)
    clean_text = clean_text.rstrip()
    return clean_text
