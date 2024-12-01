import re
import textwrap
from typing import List
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.util import ngrams
from wordcloud import WordCloud


def print_list(df:pd.DataFrame, fields:List[str]=None, sep_count:int=100, sep_char:str='-',truncate=False, wrap_width: int = 100):
    """
    Imprime una lista como string
    """
    if fields is None:
        fields = df.columns
    
    if(not truncate):
        current_max_colwidth = pd.get_option('display.max_colwidth')
        pd.option_context('display.max_colwidth', None)
        with pd.option_context('display.max_colwidth', None):
            for _, row in df.iterrows():
                for field in fields:
                    if len(str(row[field])) > wrap_width:
                        print(f"{field}:")
                        wrapped_text = textwrap.fill(str(row[field]), width=wrap_width)
                        print(wrapped_text)
                    else:
                        print(f"{field}:".ljust((wrap_width-len(str(row[field]))), ' '),row[field])
                print(sep_char * sep_count)
                print()
        pd.option_context('display.max_colwidth', current_max_colwidth)
    else:
        for _, row in df.iterrows():
                print(row[fields].to_string())
                print(sep_char * sep_count)
                print()

def remove_stop_words(tokens, stopwords):
    return [word for word in tokens if word not in stopwords]

def pos_tagging(tagged_tokens):
    return  [word for word, tag in tagged_tokens] 


def plot_frequent_terms(data, column, term_count, title, filter_terms=None, ngram_size=1, figsize=(10, 6)):
    """
    Grafica las palabras o n-gramas más frecuentes en una columna de texto.
    
    Parameters:
    - data: el conjunto de datos.
    - column: str, nombre de la columna con texto tokenizado.
    - term_count: int, cantidad de términos a mostrar.
    - title: str, título del gráfico.
    - filter_terms: list or tuple, términos que se deben excluir (opcional).
    - ngram_size: int, tamaño del n-grama (1 para palabras, 2 para bigramas, etc.).
    - figsize: tuple, tamaño de la figura del gráfico.
    """
    terms = []
    for review in data[column]:
        ngrams_list = list(ngrams(review, ngram_size)) if ngram_size > 1 else review
        terms.extend(ngrams_list)

    if filter_terms:
        terms = [term for term in terms if term != filter_terms]

    term_freq = Counter(terms)
    most_common_terms = term_freq.most_common(term_count)
    terms, counts = zip(*most_common_terms)
    
    if ngram_size > 1:
        terms = [' '.join(term) for term in terms]
    
    plt.figure(figsize=figsize)
    sns.barplot(x=list(terms), y=list(counts))
    plt.title(title)
    plt.xlabel('Términos' if ngram_size == 1 else f'{ngram_size}-Grams')
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.show()
    
def plot_wordcloud_bigrams(data, title, filter_terms=None):
    """
    Genera y grafica una nube de palabras basada en bigramas.
    
    Parameters:
    - data: str, texto completo para procesar.
    - title: str, título de la nube de palabras.
    - filter_terms: tuple, bigrama a excluir (opcional).
    """
    tokens = data.split() 
    bigrams = list(ngrams(tokens, 2))
    
    if filter_terms:
        bigrams = [bigram for bigram in bigrams if bigram != filter_terms]
    
    bigram_text = ['_'.join(bigram) for bigram in bigrams]
    bigram_freq = Counter(bigram_text)
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='cividis'
    ).generate_from_frequencies(bigram_freq)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()