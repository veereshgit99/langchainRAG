import argparse
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Debugging: Print the token to verify
print("Token loaded:", HUGGINGFACE_API_TOKEN)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Load the local embedding model
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # 384 dimensions
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.max_marginal_relevance_search(query_text, k=3, fetch_k=10)
    
    print(f"🔍 Found {len(results)} relevant documents.")

    if not results:
        print("❌ No relevant documents found.")
        return

    # Build context correctly
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print("📝 Prompt:\n", prompt)

    # Use Hugging Face API
    model = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",  # Correct model name
        task="text-generation",  # Specify the task type
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
        temperature=0.1,  # Pass directly instead of inside `model_kwargs`
        model_kwargs={
            "max_length": 500  # Pass max_length inside model_kwargs
        }
    )

    # Use invoke instead of predict
    response_text = model.invoke(prompt)

    # Extract sources correctly
    sources = [doc.metadata.get("source", None) for doc in results]
    formatted_response = f"\n✅ Response: {response_text}\n📌 Sources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()