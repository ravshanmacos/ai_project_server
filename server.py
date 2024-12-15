from flask import Flask, request, jsonify
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import os
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
openai_key = os.environ.get('OPENAI_API_KEY') 
chromadb_path = os.environ.get('CHROMA_DB_PATH')

# Initialize OpenAI and ChromaDB

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, model_name="text-embedding-3-small"
)
chroma_client = PersistentClient(path=chromadb_path)
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=openai_ef
)

client = OpenAI(api_key=openai_key)

# Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)


# Function to load text documents from a directory
def load_documents_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

# Load documents from the directory
directory_path = "essays"
documents = load_documents_from_directory(directory_path)



# Split documents into chunks

# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

chunked_documents = []

for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({
            "id": f"{doc['id']}_{i}",
            "text": chunk
        })

# Function to generate embeddings using OpenAI
def get_openai_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

# Generate embeddings for the document chunks
for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = get_openai_embedding(doc["text"])

# Upsert documents with embeddings into Chroma
for doc in chunked_documents:
    print("==== Inserting chunks into db ====")
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["text"]],
        embeddings=[doc["embedding"]]
    )

# Define the query_documents function
def query_documents(question, n_results=2):
    """
    Query the collection to retrieve the most relevant document chunks for a question.

    Args:
        question (str): The input query/question.
        n_results (int): The number of results to retrieve. Default is 2.

    Returns:
        tuple: A dictionary of raw results and a list of relevant document chunks.
    """
    try:
        # Query the collection
        results = collection.query(query_texts=question, n_results=n_results)

        # Extract relevant chunks with metadata
        relevant_chunks = []
        for sublist, doc_id_list, distance_list in zip(
            results["documents"], results["ids"], results["distances"]
        ):
            for doc, doc_id, distance in zip(sublist, doc_id_list, distance_list):
                relevant_chunks.append({
                    "text": doc,
                    "id": doc_id,
                    "distance": distance
                })

        return results, relevant_chunks

    except Exception as e:
        print(f"Error querying documents: {e}")
        return None, []

# Define a route to handle incoming queries
@app.route('/query', methods=['POST'])
def handle_query():
    """
    Endpoint to accept a question and return the relevant document chunks.
    """
    try:
        # Parse the incoming JSON payload
        data = request.get_json()
        question = data.get('question', None)
        n_results = data.get('n_results', 3)

        if not question:
            return jsonify({"error": "Question is required"}), 400

        # Query the documents
        _, relevant_chunks = query_documents(question, n_results)

        # Return the results as JSON
        return jsonify({"question": question, "results": relevant_chunks}), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Main entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)