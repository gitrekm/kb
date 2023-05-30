import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

import tiktoken
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
import os, tempfile

def estimate_cost_of_embedding(split_docs):
    enc = tiktoken.encoding_for_model("gpt-4")
    total_word_count = sum(len(doc.page_content.split()) for doc in split_docs)
    total_token_count = sum(len(enc.encode(doc.page_content)) for doc in split_docs)

    st.write(f"\nTotal word count: {total_word_count}")
    st.write(f"\nEstimated tokens: {total_token_count}")
    st.write(f"\nEstimated cost of embedding: ${total_token_count * 0.0004 / 1000}")



def main():
    st.title("Internal Knowledge Base Q&A PoC for Doc Query")

    st.header("Set Up")
    # Ask the user to enter their OpenAI API key
    API_O = st.text_input(
    ":blue[Enter Your OPENAI API-KEY :]",
    placeholder="Paste your OpenAI API key here (sk-...)",
    type="password",
    )
    #loader = DirectoryLoader('./docs/CVs/', glob='**/*.txt')
    #documents = loader.load()
    if API_O:

        uploaded_files = st.file_uploader("Choose documents", accept_multiple_files=True)
        documents = []
        st.write(uploaded_files)
        for file_name in uploaded_files:
            _, extension = os.path.splitext(file_name.name)

            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(file_name.read())
            if extension in ['.pdf']:     
                loader = PyPDFLoader(tmp_file.name)
                documents.extend(loader.load())
            elif extension in ['.doc', '.docx']:
                loader = Docx2txtLoader(tmp_file.name)
                documents.extend(loader.load())
            elif extension in ['.txt']:
                loader = TextLoader(tmp_file.name)
                documents.extend(loader.load())
        if documents:
            st.write("Documents loaded.")
            st.write("Estimated tokens and costs")
            estimate_cost_of_embedding(documents)
            embeddings = OpenAIEmbeddings(openai_api_key=API_O)
            vector_store = Chroma.from_documents(documents, embeddings)
            st.header("Query using the vector store with GPT-3 integration")
            system_template="""Use the following pieces of context to answer the users question.
            Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources.
            If you don't know the answer, just say that "I don't know", don't try to make up an answer.
            explain your choice with bullet point.
            ----------------
            {summaries}"""
            messages = [
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
            prompt = ChatPromptTemplate.from_messages(messages)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=256, openai_api_key=API_O)  # Modify model_name if you have access to GPT-4
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(),
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )

            st.header("Enter your query")
            query = st.text_input('')
            if st.button('Submit Query'):
                result = chain(query)
                print(result)
                #st.markdown(result)
                st.title('Question and Answer')
                st.write('Question: ', result['question'])
                st.write('Answer: ', result['answer'])
                st.title('Used documents for this question:')
                for i, doc in enumerate(result['source_documents'], 1):
                    with st.expander(f'Document {i}'):
                        st.text('Source: ' + doc.metadata['source'])
                        st.text('Content: ' + doc.page_content)
        else:
            st.warning("please upload the list of documents first.")
    else:
        st.warning("please enter your Openai API Key")


if __name__ == "__main__":
    main()
