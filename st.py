import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import torch
import time
import pickle
from transformers import pipeline

## FOLLOWING CODE CAN USED TO CREATE EMBEDDINGS AND SPLITS FOR THE TEXT BOOK ACCORDING TO PERSONAL MODIFICATIONS

# with open("path/ConceptsofBiology-WEB.txt") as f:   # "path" is the path for text file of the book                                                             
#     docs = f.read()

# # creates split data
# sp = create_split_data(docs)
# # creates embeddings for the whole book
# embeded_data = create_embedding(sp)

# saving and loading this data can be done as follows:
# with open('splits.pkl', 'wb') as f:
#     pickle.dump(sp, f)              
# with open('splits.pkl', 'rb') as f:
#     ssp = pickle.load(f)

##

def create_embedding(query):
    """Creates embeddings for the query"""

    model_name = 'paraphrase-mpnet-base-v2'
    model = SentenceTransformer(model_name)
    query_embedding = model.encode(query)
    return query_embedding

def retrieve_documents(query, embedded_docs, splits, top_n=3):
    """Embeds the query and based on similarity technique (cosine) selects top n documents related to it."""

    start_time = time.time()
    query_embedding = create_embedding([query])

    # cosine similarity between query and all document embeddings
    similarities = cosine_similarity(query_embedding, embedded_docs)[0]

    # indices of n most similar documents
    similarity_with_index = list(enumerate(similarities))

    # sort and select top n by similarity score
    similarity_with_index.sort(key=lambda x: x[1], reverse=True)

    top_indices = [idx for idx, _ in similarity_with_index[:top_n]]
    top_documents = [splits[i] for i in top_indices]

    end_time = time.time()
    time_elapsed = end_time - start_time
    st.write("Time for retrieving data:", time_elapsed)
    return top_documents

def get_context_and_answer(splits, query, question, embedded_docs):
    """Takes query, gets the context from retrieval, and then gets the answer from that context to the provided question"""

    start_time = time.time()
    # retrieving docs
    relevant_docs = retrieve_documents(query, embedded_docs, splits)
    context = ' '.join(relevant_docs)

    model_name = 'google-bert/bert-large-uncased-whole-word-masking-finetuned-squad'  # (another option)deepset/roberta-base-squad2

    # model for context-based answering
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input = tokenizer(question, context, return_tensors="pt")
    output = model(**input)

    # selecting best answer
    ans_start = torch.argmax(output.start_logits)
    ans_end = torch.argmax(output.end_logits)
    ans_token = input.input_ids[0, ans_start: ans_end + 1]

    # decoding the answer
    answer = tokenizer.decode(ans_token)

    end_time = time.time()
    time_elapsed = end_time - start_time
    st.write("Time for answering question based on retrieved data:", time_elapsed)
    return context, answer

def get_context_and_answer_long(splits, query, question, embedded_docs):
    """ Takes query, gets the context from retrieval, and then gets the answer from that context to the provided question """

    start_time = time.time()
    # retrieving docs
    relevant_docs = retrieve_documents(query, embedded_docs, splits)
    context = ' '.join(relevant_docs)
    
    qa_pipeline = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad") #distilbert/distilbert-base-cased-distilled-squad # deepset/roberta-base-squad2
    
    result = qa_pipeline(question=question, context=context)
    answer = result['answer']
    
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("Time for answering question based on retrieved data:", time_elapsed)
    print()
    return context, answer

## SLOWER MODEL (Can use gpt2 instead of gpt2-xl)

# def get_context_and_answer_slow(splits, query, question, embedded_docs):
#     """ Takes query gets the context from retreival and then get answer from that context to the provided question"""
#     start_time  = time.time()
#     # retreiving docs
#     relevant_docs = retrieve_documents(query, embedded_docs, splits)
#     context = ' '.join(relevant_docs)
    
#     tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")           # openai-community/gpt2-xl  openai-community/gpt2
#     model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-xl")

#     if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
#     prompt = f"Answer the following question based on the context provided. Context: {context}. Question: {question}"
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
#     gen_tokens = model.generate(
#         input_ids,
#         do_sample= False,
#         no_repeat_ngram_size=2,
#         max_length=250,
#         pad_token_id=tokenizer.pad_token_id
#     )
#     # gen_tokens = model.generate(**input_ids, pad_token_id=tokenizer.pad_token_id, max_new_tokens=200)
#     gen_text = tokenizer.batch_decode(gen_tokens)[0]
#     answer = gen_text[len(prompt):]
  
#     end_time  = time.time()
#     time_elapsed = end_time - start_time
#     print("Time for answering question based on retrieved data:",time_elapsed)
#     print()
#     print("Context:",context)
#     print()
#     print("Answer:", answer)


# Streamlit UI
st.title("PDF Question Answering System")

# Load embeddings from file
with open('embeddings.pkl', 'rb') as f:
    embedded_data = pickle.load(f)

# Load splits from file
with open('splits.pkl', 'rb') as f:
    splits = pickle.load(f)

st.write("Embeddings loaded from the backend.")

question = st.text_input("Ask a question about the PDF(few words answer type)")

if question:
    context, answer = get_context_and_answer(splits, question, question, embedded_data)
    st.write("Context:", context)
    st.write("Answer:", answer)

long_question = st.text_input("Ask a question about the PDF(long answer type)")

if long_question:
    context, answer = get_context_and_answer_long(splits, long_question, long_question, embedded_data)
    st.write("Context:", context)
    st.write("Answer:", answer)

# slow_question = st.text_input("Ask a question about the PDF(Slow but Accurate)")

# if slow_question:
#     context, answer = get_context_and_answer_slow(splits, slow_question, slow_question, embedded_data)
#     st.write("Context:", context)
#     st.write("Answer:", answer)
