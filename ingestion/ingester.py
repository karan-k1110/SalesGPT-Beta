import os
from _ctypes import sizeof
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

os.environ['PINECONE_API_KEY'] = 'xx'

def get_chunked_pdf_documents(path: str) -> list[Document]:
    loader = TextLoader(path)
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=20, separator="\n")
    return text_splitter.split_documents(document)


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-ada-002")


def ingest_document(path: str) -> bool:
    try:
        documents = get_chunked_pdf_documents(path)
        embeddings = OpenAIEmbeddings()
        print(f"Ingesting documents -> Documents -> {len(documents)}")
        to_vector_store(documents=documents, embedding=embeddings, index_name="product-catalog")
        return True
    except Exception as e:
        return False


def to_vector_store(documents: list[Document], embedding: OpenAIEmbeddings, index_name: str) -> bool:
    try:
        Pinecone.from_documents(documents=documents, embedding=embedding, index_name=index_name)
        return True
    except Exception as e:
        print("The error is: ", e)
        return False


if __name__ == "__main__":
    pdf_path = "/Users/karan/Developer/Personal/projects/SalesGPT-Beta/ingestion/product_catalog"
    ingest_document(pdf_path)
