import dataLoader
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from logs import get_logger
from uuid import uuid4

logging=get_logger(__name__)
logging.info("Initializing embeddings")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

index=faiss.IndexFlatL2(len(embeddings.embed_query("Parameter Efficient Fine-Tuning ")))
vector_store = FAISS(
    index=index,
    embedding_function=embeddings,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
def embed_text(allContent):
    for docs in allContent:
        vector_store.add_documents(
            documents=[docs],
            embedding=embeddings,
        )
    logging.info("Embedded all content into vector store")
    vector_store.save_local("faiss_index")
    # vector_store = FAISS.load_local("faiss_index", embeddings)
    logging.info("VectorDB is created successfully")
def main():
    allContent=dataLoader.main()
    embed_text(allContent)
    # return allContent

if __name__ == "__main__":
    main()