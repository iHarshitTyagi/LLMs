import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA

import os, tempfile


#Defining a function called 'Load documents'
# We can load mutliple  type of file 

def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    
    if extension == '.pdf':
        print(f'Loading {file}')
        loader = PyPDFLoader(file)

    elif extension == '.docx':
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)

    elif extension == '.txt':
        print(f'Loading {file}')
        loader = TextLoader(file)

    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data





# chunk data function 

def chunk_data(data, chunk_size=500):#chunk_size=500, chunk_overlap=50
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    chunks = text_splitter.split_documents(data)
    return chunks
  

## Creating embedding and storing in vector store

def create_embeddings(chunks):
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embeddings)
    #returning vector store
    return vector_store

def ask_and_get_answer(vector_store, q):

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    vicuna_model_path = "C:\\Users\\Desktop\\notebook_pdf\\stable-vicuna-13B.ggmlv3.q4_1.bin"

    
    llm_model_path = vicuna_model_path

    llm1 = LlamaCpp(
        model_path=llm_model_path, callback_manager=callback_manager, verbose=True
        )

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    chain = RetrievalQA.from_chain_type(llm=llm1, chain_type="stuff", retriever=retriever)
    
    answer = chain.run(q)
    return answer

def clear_history():
    if 'history ' in st.session_state:
        del st.session_state['history']




if __name__=="__main__":
    import os
    from dotenv import load_dotenv,find_dotenv
    load_dotenv(find_dotenv(),override=True)


    # Streamlit app

    st.subheader('Generative Q&A with LangChain & Pinecone')
    with st.sidebar:
        api_key=st.text_input('HuggingFaceAPI',type='password')
        if api_key:
            os.environ['HUGGINGFACEHUB_API_TOKEN']=api_key

        
        uploader_file=st.file_uploader('Upload source document:',type=['pdf','docx','txt'],label_visibility="collapsed")
        # query = st.text_input("Enter your query")

        chunk_size=st.number_input('Chunk size',min_value=100,max_value=2048,value=512,on_change=clear_history)
        k=st.number_input('k',min_value=1,max_value=20,value=3,on_change=clear_history)

        add_data=st.button('Add your Data Here Please!',on_click=clear_history)

        if uploader_file and add_data:
            with st.spinner('Reading ,Chunking and Embedding file ......'):

                bytes_data=uploader_file.read()
                file_name=os.path.join('./',uploader_file.name)

                with open(file_name,'wb') as f:
                    f.write(bytes_data)

                data=load_document(file_name)

                chunks=chunk_data(data, chunk_size=500)
                st.write(f'Chunk Size :{chunk_size},Chunks:{len(chunks)}')

                vector_store=create_embeddings(chunks)


                st.session_state.vs=vector_store
                st.sucess('File uploaded, chunked and embedded sucessfully!..')
    

    q=st.text_input('Ask a question about your doucments..')

    if q:
        if 'vs' in st.session_state.vs:
            vector_store=st.session_state.vs
            
            st.write(f'k: {k} ')
            answer=ask_and_get_answer(vector_store,q,k)

            st.text_area('LLM Answer',value=answer)
    
            st.divider()
            if 'history' not in st.session_state:
                st.session_state.history=''

            value =f'Q: {q}\nA: {answer}'
            st.session_state.history=f'{value} \n {"-"*100}\n {st.session_state.history}'
            h=st.seesion_state.history

            st.text_area(label='Chat History',value=h,key='history',height=400)
