import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from .env file
TOP_K = int(os.getenv("TOP_K", "3"))  # Number of similar documents to retrieve
MODEL_NAME = os.getenv("OLLAMA_MODEL", "phi3")  # Ollama model name
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # Embedding model

def load_vectorstore(index_path="faiss_index", embedding_model=None):
    """Load FAISS index from disk"""
    if embedding_model is None:
        embedding_model = EMBEDDING_MODEL
    
    if not os.path.exists(index_path):
        print(f"Error: Index not found at '{index_path}'")
        print("Please run indexer.py first to create the index!")
        return None
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = FAISS.load_local(
            index_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore
    except Exception as e:
        print(f"Error loading index: {e}")
        return None

def get_similar_documents(vectorstore, query, k=TOP_K):
    """Find top K similar documents with scores"""
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
    return docs_and_scores

def format_docs(docs):
    """Format documents for the prompt"""
    return "\n\n".join([doc.page_content for doc in docs])

def generate_answer(llm, query, context_docs):
    """Generate answer using LLM with retrieved context"""
    # Build context from documents
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question based on the provided context from PDF documents. If the context doesn't contain relevant information, say 'I don't know.' Keep your response concise and informative."),
        ("user", "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
    ])
    
    # Create chain
    chain = (
        {"context": lambda x: context, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Get response
    response = chain.invoke(query)
    return response

def main():
    """Simple chatbot: query → retrieve → show scores → generate answer"""
    print("=" * 60)
    print("PDF RAG Chatbot (Fully Local)")
    print("=" * 60)
    
    # Check if Ollama is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            raise Exception("Ollama not responding")
    except Exception as e:
        print(f"\n⚠️  Warning: Could not connect to Ollama!")
        print("Please make sure Ollama is installed and running.")
        print("\nTo install Ollama:")
        print("  - macOS: brew install ollama")
        print("  - Linux: curl -fsSL https://ollama.com/install.sh | sh")
        print("  - Windows: Download from https://ollama.com")
        print("\nThen start Ollama and pull a model:")
        print(f"  ollama pull {MODEL_NAME}")
        sys.exit(1)
    
    # Get index path
    index_path = sys.argv[1] if len(sys.argv) > 1 else "faiss_index"
    
    # Load vectorstore
    print(f"Loading index from '{index_path}'...")
    vectorstore = load_vectorstore(index_path)
    if vectorstore is None:
        sys.exit(1)
    
    # Initialize LLM (Ollama)
    print(f"Initializing Ollama model '{MODEL_NAME}'...")
    try:
        llm = OllamaLLM(model=MODEL_NAME, temperature=0.1)
        # Test connection
        llm.invoke("test")
        print(f"✓ Connected to Ollama")
    except Exception as e:
        print(f"Error: Could not load model '{MODEL_NAME}'")
        print(f"Make sure you've pulled the model: ollama pull {MODEL_NAME}")
        print(f"Available models: ollama list")
        sys.exit(1)
    
    print(f"\n✓ Ready! Using {MODEL_NAME}, retrieving top {TOP_K} matches")
    print("Type 'exit' to quit")
    print("-" * 60)
    
    # Main loop
    while True:
        print("\n")
        query = input("You: ").strip()
        
        if not query:
            continue
        
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        try:
            # Step 1: Find similar documents
            docs_with_scores = get_similar_documents(vectorstore, query, k=TOP_K)
            
            # Step 2: Display matches with scores and text snippets
            print(f"\n🔍 Top {TOP_K} Similar Documents:")
            for i, (doc, score) in enumerate(docs_with_scores, 1):
                source = doc.metadata.get("source", "Unknown")
                # Lower score is better for cosine similarity
                status = "✅" if score < 0.4 else "⚠️" if score < 0.5 else "❌"
                
                # Get text snippet (first 200 chars)
                text_snippet = doc.page_content[:200].replace('\n', ' ')
                if len(doc.page_content) > 200:
                    text_snippet += "..."
                
                print(f"\n  {i}. {source} - Score: {score:.3f} {status}")
                print(f"     \"{text_snippet}\"")
            
            # Step 3: Generate answer using LLM
            context_docs = [doc for doc, score in docs_with_scores]
            print("\nBot: ", end="", flush=True)
            answer = generate_answer(llm, query, context_docs)
            print(answer)
        
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()