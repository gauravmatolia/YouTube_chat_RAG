# YouTube Video Q&A with RAG (Retrieval-Augmented Generation)

This project demonstrates how to build a simple Question-Answering system for a YouTube video using a Retrieval-Augmented Generation (RAG) approach. It fetches the transcript of a YouTube video, processes it, creates vector embeddings, and then uses a Large Language Model (LLM) to answer questions based on the retrieved relevant parts of the transcript.

## Setup

### 1. Install Dependencies

```python
!pip install -q langchain-community langchain_google_genai langchain_huggingface \ faiss-cpu tiktoken python-dotenv
!pip install -U sentence-transformers
!pip install -U langchain_FAISS
!pip install youtube-transcript-api
```

### 2. Import Libraries

```python
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import os
from google.colab import userdata
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
```

## Workflow

### 1. Fetch YouTube Transcript

The `YouTubeTranscriptApi` is used to fetch the English transcript of a specified YouTube video. The `video_id` variable should be set to the ID of the desired YouTube video.

```python
video_id = "LPZh9BOjkQs"

try:
   transcript_list = YouTubeTranscriptApi().fetch(video_id=video_id, languages=["en"])
   transcript = " ".join(chunk.text for chunk in transcript_list)
   print(transcript)
except TranscriptsDisabled:
  print("Some error occured")
```

### 2. Text Splitting

The fetched transcript is too long to be processed by an LLM in one go. Therefore, it's split into smaller, overlapping chunks using `RecursiveCharacterTextSplitter`. This helps maintain context between chunks.

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
print(len(chunks))
```

### 3. Vector Embeddings

Each text chunk is converted into a numerical vector (embedding) using a pre-trained `HuggingFaceEmbeddings` model (`sentence-transformers/all-MiniLM-L6-v2`). These embeddings capture the semantic meaning of the text.

```python
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### 4. Create Vector Store

The embeddings are stored in a FAISS vector store, which allows for efficient similarity search. This store will be used to retrieve relevant chunks based on a user's query.

```python
vector_store = FAISS.from_documents(chunks, embedding)
```

### 5. Initialize Retriever

A retriever is created from the FAISS vector store. When a question is posed, this retriever will find the `k` most similar document chunks to the question's embedding.

```python
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k":4})
```

### 6. Set up Large Language Model (LLM)

The `ChatGoogleGenerativeAI` model (`gemini-2.5-flash-lite`) is used as the core LLM for generating answers. An API key for Google Gemini is required and should be stored securely.

```python
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)
```

### 7. Define Prompt Template

A `PromptTemplate` is used to structure the input to the LLM, ensuring it receives the context (retrieved document chunks) and the user's question in a clear format. The LLM is instructed to answer *only* from the provided context.

```python
prompt = PromptTemplate(
    template="""
    You are a helpful AI assitant,
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    Context: {context}
    Question: {question}
    """,
    input_variables=['context', 'question']
)
```

### 8. Build the RAG Chain

Langchain's `Runnable` components are used to create a pipeline (chain) that orchestrates the RAG process:
1. **Parallel Chain**: Fetches relevant document chunks using the retriever and formats them into a single string. It also passes the original question.
2. **Prompt**: Formats the context and question according to the `PromptTemplate`.
3. **LLM**: Generates the answer based on the formatted prompt.
4. **Output Parser**: Parses the LLM's output into a string.

```python
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser
```

### 9. Interactive Q&A Loop

Finally, an interactive loop allows the user to ask questions and receive answers generated by the RAG system. The loop continues until the user types 'quit'.

```python
while True:
  user_input = input("\nEnter your question: ")
  if(user_input == 'quit'):
    print("Exiting the code")
    break
  else:
    rag_ouput = main_chain.invoke(user_input)
    print(f"Answer: {rag_ouput}\n")
```
