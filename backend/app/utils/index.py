import logging
from  llama_index.core.indices.vector_store.base import VectorStoreIndex, ServiceContext
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import StorageContext, load_index_from_storage
import weaviate
import box
import yaml

from llama_index.core  import (
    VectorStoreIndex,
    ServiceContext
)


def get_index():
    """
    Retrieves or creates a VectorStoreIndex for searching documents based on their semantic similarity.

    Returns:
        A VectorStoreIndex object loaded from configuration and Weaviate.
    """
    logger = logging.getLogger("uvicorn")

    # load the existing index
    logger.info(f"Loading index.")
    
    # Import configuration specified in config.yml
    with open('config.yml', 'r', encoding='utf8') as ymlfile:
        cfg = box.Box(yaml.safe_load(ymlfile))

    logger.info("Connecting to Weaviate")
    resource_owner_config = weaviate.AuthClientPassword(
    username="lyreachsak@yahoo.com",
    password="Reachsak1122@@",
)
    client = weaviate.Client("http://localhost:8080")

    logger.info("Loading Ollama...")
    llm = Ollama(model="llama3.1:8b-instruct-q8_0", temperature=0.3)
    # llm = Ollama(model="dolphin-mixtral:8x7b-v2.7-q3_K_L", temperature=0.3)

    logger.info("Loading embedding model...")
    embeddings = load_embedding_model(model_name=cfg.EMBEDDINGS)

    logger.info("Loading index...")
    index = load_index(cfg.CHUNK_SIZE, llm, embeddings, client, cfg.INDEX_NAME)

    return index

def load_embedding_model(model_name):
    """
    Creates a LangchainEmbedding wrapper around a HuggingFace embedding model.

    Args:
        model_name: Name of the pre-trained HuggingFace model to use for embeddings.

    Returns:
        A LangchainEmbedding object for generating sentence embeddings.
    """
    embeddings = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name=model_name)
    )
    return embeddings

def load_index(chunk_size, llm, embed_model, weaviate_client, index_name):
    """
    Creates a VectorStoreIndex object using a WeaviateVectorStore and ServiceContext.

    Args:
        chunk_size: Number of documents processed per batch for indexing.
        llm: Ollama instance for generating query representations.
        embed_model: LangchainEmbedding instance for generating document embeddings.
        weaviate_client: Weaviate client connected to the target Weaviate instance.
        index_name: Name of the Weaviate index to use for storing document vectors.

    Returns:
        A VectorStoreIndex object configured for Weaviate and semantic search.
    """
    service_context = ServiceContext.from_defaults(
        chunk_size=chunk_size,
        llm=llm,
        context_window=350,
        embed_model=embed_model
    )

    vector_store = WeaviateVectorStore(weaviate_client=weaviate_client, index_name=index_name)

    # index = load_index_from_storage( storage_context=storage_context,index_name=index_name)
    
    storage_context = StorageContext.from_defaults( vector_store=vector_store, persist_dir="./storage")
    
    
    
    

    index = VectorStoreIndex.from_vector_store(
        vector_store, service_context=service_context,storage_context=storage_context
    )

    
    
    return index
