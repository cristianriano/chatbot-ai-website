# Chatbot with AI about a Website

This is baed on the openai tutorial: [How to build an AI that can answer questions about your website](https://platform.openai.com/docs/tutorials/web-qa-embeddings)

It goes step by step
1. Creating a crawler from scratch ([crawler.py](./crawler.py))
2. Cleaning the text from the crawler ([csv_formatter.py](./csv_formatter.py))
3. Splitting the text into sentences by the number of tokens (common sequences of characters found in text) ([tokenizer.py](./tokenizer.py))
4. Creating the embeddings with OpenAI [embeddings.py](./embeddings.py)
5. Creating the API to answer questions

## Setup

1. [OPTIONAL] Create and activate a virtual environment
```
python -m venv .venv
. .venv/bin/activate
```
2. Set the API Key by copying the `.env` and setting the values
```
cp .env.example .env
```
3. Install the dependencies
```
pip install -r requirements.txt
```
