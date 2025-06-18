from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_splitter import SentenceSplitter

def split_langchain(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    return splitter.split_text(text)


def split_ssplitter(text):
    splitter = SentenceSplitter(language='en')
    sentences = splitter.split(text)
    return [s for s in sentences if s.strip()]
