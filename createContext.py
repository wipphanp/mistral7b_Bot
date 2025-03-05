from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import warnings
warnings.filterwarnings("ignore")


# Step1 : Load Raw PDFs

#Extract data from the PDF

DATA_PATH = './data/'

def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()

    return documents

documents = load_pdf(data=DATA_PATH)
print("\n \nLength of PDF Documents: ", len(documents))

# Step2 : Create chunks   
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks

text_chunks = text_split(extracted_data=documents)
print("length of my chunk:", len(text_chunks))

# Step3 : Create Vector Embeddings

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# Step4 : Store Embeddings in FAISS
DB_FAISS_PATH="vectorestore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)   
db.save_local(DB_FAISS_PATH)
