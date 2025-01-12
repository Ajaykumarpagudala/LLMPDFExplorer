import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
import pickle
# from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
import os

# Sidebar content
with st.sidebar:
    st.set_page_config(page_title="PDFAI", page_icon="pdf logo.png")
    st.image('pdf logo.png')
    # st.title('PDF Explorer powered by LLM')
    st.markdown("""
    <style>
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }

    .title {
        animation: fadeIn 2s ease-in-out;
    }
    </style>
""", unsafe_allow_html=True)

# Displaying the title with the transition class
    st.markdown('<h1 class="title">𝙋𝘿𝙁 𝙀𝙭𝙥𝙡𝙤𝙧𝙚𝙧 𝙥𝙤𝙬𝙚𝙧𝙚𝙙 𝙗𝙮 𝙇𝙇𝙈</h1>', unsafe_allow_html=True)
    st.markdown('''
    ## About
    Welcome to **PDF Explorer**, your AI-powered solution for exploring and interacting with PDF documents effortlessly. Powered by state-of-the-art Large Language Models (LLMs), this tool enables you to:

    - **Search and Query**: Ask questions from any PDF, and get instant, accurate responses.
    - **Explore Documents**: Seamlessly navigate through PDF content with intelligent summarization and context retrieval.
    - **Custom Answers**: Get precise answers based on the content of the PDF you're exploring.

    Whether you're analyzing reports, manuals, or research papers, **PDF Explorer** streamlines the process of extracting valuable insights, making it your perfect companion for document exploration.
    
    Start by uploading your PDF and ask any questions you have – the LLM will guide you through the content!
''')
    
    add_vertical_space(5)
    st.write('Made by ajaykumarpagudala')

# load_dotenv()

def main():
    st.header("𝙀𝙭𝙥𝙡𝙤𝙧𝙚 𝙒𝙞𝙩𝙝 𝙇𝙇𝙈 𝙑𝙄𝘾𝙐𝙉𝘼")
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        # Extract text from PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into chunks
        textsplitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = textsplitter.split_text(text=text)

        # Embeddings and vectorstore
        pdf_name = pdf.name[:-4]
        if os.path.exists(f"{pdf_name}.pkl"):
            with open(f"{pdf_name}.pkl", "rb") as f:
                Vectorstore = pickle.load(f)
            st.write('Embeddings loaded successfully')
        else:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            Vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{pdf_name}.pkl", "wb") as f:
                pickle.dump(Vectorstore, f)
            st.write('Embeddings computed successfully')

        # Handle query input
        # Handle query input
    query = st.text_input('Ask some questions from the PDF')
    if query:
        docs = Vectorstore.similarity_search(query=query, k=1)
        context = "\n".join([doc.page_content for doc in docs])  # Combine retrieved document content
        prompt = f"Context: {context}\n\nQuestion: {query}"  # Prepare the prompt
        llm = Ollama(model="vicuna")  # Replace "vicuna" with the desired model
        response = llm.generate(prompts=[prompt])  # Pass the prompt as a list
        str=''
        for i in response.generations[0]:
            for j in i:
                st.write(j[1])
                break
if __name__ == '__main__':
    main()
