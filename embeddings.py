import openai
import pandas as pd
import os
from dotenv import load_dotenv

MODEL = 'text-embedding-ada-002'

# Load your API key from an environment variable
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


df = pd.read_csv('processed/shortened.csv', index_col=0)
df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine=MODEL)['data'][0]['embedding'])

df.to_csv('processed/embeddings.csv')
print(df.head())
