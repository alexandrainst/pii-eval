"""
This section covers functionality for (down)loading Named Entity
Recognition data sets.
"""

import os
import json
import pyconll
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import ssl


def download_unzip(url_zip: str,
                   dir_extract: str) -> str:
    """Download and unzip a ZIP archive to folder.

    Loads a ZIP file from URL and extracts all of the files to a
    given folder. Does not save the ZIP file itself.

    Args:
        url_zip (str): URL to ZIP file.
        dir_extract (str): Directory where files are extracted.

    Returns:
        str: a message telling, if the archive was succesfully
        extracted. Obviously the files in the ZIP archive are
        extracted to the desired directory as a side-effect.
    """

    # suppress ssl certification
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    print(f'Reading {url_zip}')
    with urlopen(url_zip, context=ctx) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(dir_extract)

    return f'archive extracted to {dir_extract}'


def download_dane_data(dir: Path = None) -> str:
    """Download DaNE data set.

    Downloads the 'DaNE' data set annotated for Named Entity
    Recognition developed and hosted by
    [Alexandra Institute](https://github.com/alexandrainst/danlp/blob/master/docs/docs/datasets.md#dane).

    Args:
        dir (str, optional): Directory where DaNE datasets will be saved. If no directory is provided, data will be saved to a hidden folder '.dane' in your home directory.

    Returns:
        str: a message telling, if the archive was in fact
        succesfully extracted. Obviously the DaNE datasets are
        extracted to the desired directory as a side-effect.

    Examples:
        >>> download_dane_data()
        >>> download_dane_data(dir = 'DaNE')

    """
    # set to default directory if nothing else has been provided by user.
    if dir is None:
        dir = Path("data/raw/dane")

    return download_unzip(
        url_zip='http://danlp-downloads.alexandra.dk/datasets/ddt.zip',
        dir_extract=dir)


def get_dane_data(split: str = 'train',
                  limit: int = None,
                  dir: Path = None) -> dict:
    """Load DaNE data split.

    Loads a single data split from the DaNE data set kindly hosted
    by [Alexandra Institute](https://github.com/alexandrainst/danlp/blob/master/docs/docs/datasets.md#dane).

    Args:
        split (str, optional): Choose which split to load. Choose
            from 'train', 'dev' and 'test'. Defaults to 'train'.
        limit (int, optional): Limit the number of observations to be
            returned from a given split. Defaults to None, which implies
            that the entire data split is returned.
        dir (str, optional): Directory where data is cached. If set to
            None, the function will try to look for files in '.dane' folder in home directory.

    Returns:
        dict: Dictionary with word-tokenized 'sentences' and named
        entity 'tags' in IOB format.

    Examples:
        Get test split
        >>> get_dane_data('test')

        Get first 5 observations from training split
        >>> get_dane_data('train', limit = 5)

    """
    assert isinstance(split, str)
    splits = ['train', 'dev', 'test']
    assert split in splits, f'Choose between the following splits: {splits}'

    # set to default directory if nothing else has been provided by user.
    if dir is None:
        dir = Path("data/raw/dane")
    assert dir.exists(
    ), f'Directory {dir} does not exist. Try downloading DaNE data with download_dane_data()'

    file_path = os.path.join(dir, f'ddt.{split}.conllu')
    assert os.path.isfile(
        file_path), f'File {file_path} does not exist. Try downloading DaNE data with download_dane_data()'

    split = pyconll.load_from_file(file_path)

    sentences = []
    entities = []
    texts = []

    for sent in split:
        sentences.append([token.form for token in sent._tokens])
        entities.append([token.misc['name'].pop() for token in sent._tokens])
        texts.append(sent.text)

    if limit is not None:
        sentences = sentences[:limit]
        entities = entities[:limit]
        texts = texts[:limit]

    return {'texts': texts, 'sentences': sentences, 'tags': entities}


def _tokens_to_offsets(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    """Align tokens to their character offsets in text."""
    offsets = []
    cursor = 0
    text_len = len(text)

    for token in tokens:
        while cursor < text_len and text[cursor].isspace():
            cursor += 1
        start = text.find(token, cursor)
        if start == -1:
            raise ValueError(f"Token '{token}' not found in text segment.")
        end = start + len(token)
        offsets.append((start, end))
        cursor = end

    return offsets


def _bio_to_spans(text: str,
                  offsets: list[tuple[int, int]],
                  tags: list[str]) -> list[dict]:
    """Convert BIO tags aligned with offsets to span dictionaries."""
    spans = []
    active_label = None
    active_start = None
    active_end = None

    for (start, end), tag in zip(offsets, tags):
        if not tag or tag == 'O':
            if active_label:
                spans.append({
                    'label': active_label,
                    'start': active_start,
                    'end': active_end,
                    'text': text[active_start:active_end]
                })
                active_label = None
            continue

        if '-' in tag:
            prefix, label = tag.split('-', 1)
        else:
            prefix, label = 'B', tag

        if prefix == 'I' and active_label == label:
            active_end = end
            continue

        if active_label:
            spans.append({
                'label': active_label,
                'start': active_start,
                'end': active_end,
                'text': text[active_start:active_end]
            })

        active_label = label
        active_start = start
        active_end = end

    if active_label:
        spans.append({
            'label': active_label,
            'start': active_start,
            'end': active_end,
            'text': text[active_start:active_end]
        })

    return spans


path = Path("data/raw/dane")
if not path.exists():
    download_dane_data()
input_data = get_dane_data()

texts = input_data['texts']
sentences = input_data['sentences']
tags = input_data['tags']

documents = []
for idx, (text, tokens, bio_tags) in enumerate(zip(texts, sentences, tags)):
    offsets = _tokens_to_offsets(text, tokens)
    spans = _bio_to_spans(text, offsets, bio_tags)
    documents.append({
        'document': idx,
        'text': text,
        'spans': spans
    })

output_path = Path("data/span/dane.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open('w', encoding='utf-8') as f:
    json.dump(documents, f, ensure_ascii=False, indent=4)
