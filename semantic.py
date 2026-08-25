from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from yaspin import yaspin
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma
# from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import json
from langchain_core.documents import Document



llm = ChatOllama(
    base_url="",
    model="qwen3-vl:30b",
    temperature=0.9
)

embeddings = OllamaEmbeddings(
    base_url="",
    model="qwen3-embedding:8b")

#파일 로드 
file_path = "nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)  #text를 어떻게 분리하는지 

# embedding_model = OpenAIEmbeddings(all_splits)

# vector_store = InMemoryVectorStore(embeddings) #vector store을 파일로 하는거
# vector_store.add_documents(all_splits)

vector_store = Chroma.from_documents(    #vector db로 저장하기 위함 
    documents=all_splits,
    embedding=embeddings,
    persist_directory="chroma_store"
)


vector_store = Chroma(
    persist_directory="./chroma_store",
    embedding_function=embeddings
)

# vector_store.get()

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 1}) # retriever 옵션을 여러개 넣기 


retrieved_documents = retriever.invoke(input("YOU : "))


data = [
    {
        "page_content": doc.page_content.splitlines(),
        "metadata": doc.metadata,
    }
    for doc in retrieved_documents
]

with open("vector_store.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# print(retrieved_documents[0].page_content)

# print(type(retriever))
print("완료")
