# Flask Server for Document Querying

This project sets up a Flask server to process documents, generate embeddings, and query them using OpenAI and ChromaDB.

---

## Setup Guide

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository

First, clone the repository to your local machine:
```bash
git clone <your-repo-url>
cd <your-repo-folder>

### 1. Create a Virtual Environment
#### On Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate

#### On Windows:
```bash
python -m venv venv
.\venv\Scripts\activate

### 2. Install Required Packages
```bash
pip install -r requirements.txt

### 3. Add Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key_here

### 4. Run Server
```bash
python app.py
