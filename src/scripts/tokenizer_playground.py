import spacy
from spacy.cli import download
from nltk.tokenize import TreebankWordTokenizer


spacy_tag = "da_core_news_sm"
download(spacy_tag)
nlp = spacy.load(spacy_tag)
tokenizer = TreebankWordTokenizer()

text_1 = 'For otte millioner pund er byttet "Living Legend", hvor man bliver ført gennem hele historien med billeder, lyd, tåge, lyn, torden, storm og andre effekter.'
text_2 = 'Han sagde "hej verden" i dag.'

words_spacy_1 = [tok.text for tok in nlp(text_1)]
words_nltk_1 = tokenizer.tokenize(text_1)

words_spacy_2 = [tok.text for tok in nlp(text_2)]
words_nltk_2 = tokenizer.tokenize(text_2)

pass
