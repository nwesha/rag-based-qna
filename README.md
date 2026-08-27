# PDF Question Answering System

This project is a Streamlit-based application designed to answer questions related to the content of a PDF file. The application processes a textbook, creates embeddings, retrieves relevant sections, and generates answers to user queries using different models.

## Features

- **Text Embedding:** Create embeddings for the text from the provided PDF.
- **Document Retrieval:** Retrieve relevant documents based on the cosine similarity of the embeddings.
- **Question Answering:** Use pre-trained models to answer questions based on the retrieved documents.
- **User Interface:** Streamlit-based UI to interact with the system.

## Setup Guide

### Prerequisites

Ensure you have Python installed on your machine. You'll also need to install the required Python packages listed in `requirements.txt`.

### Installation

1. Clone the repository or download the project files.
2. Navigate to the project directory.
3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

### Running the Application

1. **Prepare the Data:**

   Ensure you have the necessary data files (`ConceptsofBiology-WEB.txt`, `embeddings.pkl`, `splits.pkl`). If not, you need to create them by processing the PDF as per the commented code in `st.py`.

2. **Start the Streamlit App:**

    ```sh
    streamlit run st.py
    ```

### Usage

- **Embeddings Preparation:**

  If you need to create embeddings and splits, use the following code snippets in `st.py` (uncomment and modify the paths as needed):

  ```python
  with open("path/to/ConceptsofBiology-WEB.txt") as f:
      docs = f.read()

  sp = create_split_data(docs)
  embeded_data = create_embedding(sp)

  with open('splits.pkl', 'wb') as f:
      pickle.dump(sp, f)

  with open('embeddings.pkl', 'wb') as f:
      pickle.dump(embeded_data, f)
