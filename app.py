import ollama
import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import re


# Call the Ollama model to generate a response 
response = ollama.chat(
    model="deepseek-r1:1.5b",
    messages=[
        {"role": "user", "content": "What is the capital of Egypt?"},
    ],
)

#Print the response
print(response['message']['content'])


# Load a PDF document
def load_pdf(pdf_path):
    """Load a PDF document, extract text, split it into chunks, and create a vector store."""

    if pdf_path is None:
        return None, None, None

    try:
        loader = PyMuPDFLoader(pdf_path) # Load the PDF document
        data = loader.load() # Extract the text from the PDF
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
        # Create a vector store from the chunks
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        retriever = vectorstore.as_retriever() # Create a retriever from the vector store
        
        return text_splitter, vectorstore, retriever
    except Exception as e:
        print(f"PDF processing error: {str(e)}")
        return None, None, None


# Combining retrieved documents chunks
def combine_docs(docs):
    """Combine the retrieved document chunks into a single string."""
    return "\n\n".join([doc.page_content for doc in docs])


# Querying DeeSeek model using Ollama
def ollama_llm(question, context):
    """Query the DeeSeek model using Ollama with the provided question and context."""

    formatted_prompt = f"Question: {question} \n\n Context: {context}"
    response = ollama.chat(
        model="deepseek-r1:1.5b",
        messages=[{"role": "user", "content": formatted_prompt}],
    )
    response_content = response['message']['content']
    # Clean up the response content by removing <think> tags
    final_answer = re.sub(r'<think>.*?<think>',
                          '',
                           response_content,
                           flags=re.DOTALL).strip()
    return final_answer


# Building the RAG pipeline
def rag_chin(question, text_splitter, vectorstore, retriever):
    """Run the RAG pipeline to answer a question using the provided text splitter, vector store, and retriever."""

    try:
        retrieved_docs = retriever.invoke(question)  # Retrieve relevant document chunks
        formatted_context = combine_docs(retrieved_docs)  # Combine the retrieved chunks
        return ollama_llm(question, formatted_context)  # Query the Ollama model with the question and context
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Create a funtion that perform the logic expected by the Chatbot
def ask_question(pdf_path, question):
    """Process the PDF and answer the question using the RAG pipeline."""

    if not pdf_path:
        return "Please upload a PDF document first."

    text_splitter, vectorstore, retriever = load_pdf(pdf_path)  # Load the PDF and create the vector store
    if not all([text_splitter, vectorstore, retriever]):
        return "Failed to process PDF. Please check the file and try again."
    result = rag_chin(question, text_splitter, vectorstore, retriever)  # Run the RAG pipeline
    return result  # Return the answer to the question


# Create a Gradio interface for the RAG pipeline
interface = gr.Interface(
    fn=ask_question,
    inputs=[
        gr.File(label="Upload PDF Document (optional)", type="filepath"),
        gr.Textbox(label="Ask a question about the document"),
    ],
    outputs=gr.Textbox(),
    title="Ask a Question about your PDF Document",
    description="Use DeepSeek R1 to answer questions about your PDF documents. ",
)
# Launch the Gradio interface
interface.launch(
    show_error=True,
    share=True,  # Critical fix for localhost issues
    server_name="0.0.0.0",  # Optional: Allows network access
    server_port=7860        # Optional: Specify port
)
