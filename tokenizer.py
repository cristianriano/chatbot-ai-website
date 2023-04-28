import tiktoken
import pandas as pd
import matplotlib.pyplot as plt

from pandas import DataFrame

# Newest models can handle inputs up to 8191 tokens
MAX_TOKENS = 500

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")


def get_tokens_distribution(df: DataFrame):
    if 'n_tokens' not in df.columns:
        # Tokenize the text and save the number of tokens to a new column
        df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    # Visualize the distribution of the number of tokens per row using a histogram
    df.n_tokens.hist()
    plt.show()
    return df


# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens=MAX_TOKENS):
    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks


def shorten_tokens(df: DataFrame):
    shortened = []

    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > MAX_TOKENS:
            shortened += split_into_many(row[1]['text'])

        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append(row[1]['text'])

    return shortened


df = pd.read_csv('processed/scraped.csv', index_col=0)
df.columns = ['title', 'text']

df = get_tokens_distribution(df)

if df.n_tokens.max() > MAX_TOKENS:
    result = shorten_tokens(df)
    df = pd.DataFrame(result, columns=['text'])
    df = get_tokens_distribution(df)

df.to_csv('processed/shortened.csv')