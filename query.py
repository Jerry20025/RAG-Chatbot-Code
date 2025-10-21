import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from logs import get_logger

logger = get_logger(__name__)
logger.info("Loading vector DB from local storage")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
print(len(vector_store.index_to_docstore_id))
logger.info("Vector DB loaded successfully")
def queryDB(queryText):
    try:
        logger.info("Querying the vector DB")
        docs = vector_store.max_marginal_relevance_search(queryText, k=5)
        logger.info(f"Retrieved {len(docs)} documents from vector DB")
        return docs
    except Exception as e:
        logger.error(f"Error querying vector DB: {e}")
        return []

def main():
    sample_query = "how qlora and lora differ in parameter efficient fine tuning and how is this affecting the AI LLM?"
    results = queryDB(sample_query)
    with open("query_results.txt", "w", encoding="utf-8") as f:
        for i, doc in enumerate(results):
            f.write(f"Document {i+1}:\n{doc.page_content}\n\n")

if __name__ == "__main__":
    main()