#  python C:\session_bot\chat_session.py
#  http://127.0.0.1:5000/ (Press CTRL+C to quit)

import os
from flask import Flask, session, render_template, jsonify, request, redirect, url_for
import openai
from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from flask import session

whole_path = r'C:\session_bot'
os.chdir(whole_path)

OPENAI_API_KEY = "#############"

OPENAI_EMBEDDING_MODEL_DEP_NAME = "textembedding"
OPENAI_EMBEDDING_MODEL_NAME = 'text-embedding-ada'

OPENAI_GPT_MODEL_DEP_NAME = "gpt4"
OPENAI_GPT_MODEL_NAME = "gpt-4"

OPENAI_API_VERSION = "2023-12-01-preview"
OPENAI_API_BASE = "https://###############"

# Set up the Azure Chat OpenAI model
os.environ["AZURE_OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = OPENAI_API_BASE

app = Flask(__name__, template_folder='templates', static_url_path='/static')
app.secret_key = "super secret key"
app.config['TEMPLATES_AUTO_RELOAD'] = True
img_folder = os.path.join('static', 'img')
app.config['UPLOAD_FOLDER'] = img_folder

## data Data Injection into Chroma
'''
loader = TextLoader('creditcard_QA.txt', encoding='utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(documents=chunks, embedding = embeddings_model,
           persist_directory=r"C:\session_bot\data_st\chroma_db")

'''

Embeddings_model = AzureOpenAIEmbeddings(deployment = OPENAI_EMBEDDING_MODEL_DEP_NAME,
                                   model = OPENAI_EMBEDDING_MODEL_NAME,
                                   azure_endpoint = OPENAI_API_BASE,
                                   openai_api_type="azure")

def get_retriever():
    loaded_vectordb = Chroma(persist_directory = r"C:\session_bot\data_st\chroma_db", 
                             embedding_function = Embeddings_model)
    retriever = loaded_vectordb.as_retriever(search_type="mmr", k = 5)
    return retriever

def get_chunks(question):
  loaded_vectordb = Chroma(persist_directory= r"C:\session_bot\data_st\chroma_db", 
                           embedding_function = Embeddings_model)
  docs = loaded_vectordb.max_marginal_relevance_search(question, k = 4)
  chunks = ' '.join([chunk.page_content for chunk in docs])
  return chunks

@app.route('/')
def main_page():
    return render_template('main_page.html')

@app.route('/chatbot_window.html')
def chatbot_window():
    return render_template('chatbot_window.html')


@app.route('/reset_history', methods=['GET'])
def reset_history():

    session.clear()
   
    return redirect(url_for('chat')) 
    
@app.route('/')
def chat():

    session.clear()

    return render_template('chat.html')


@app.route('/send', methods=['POST'])
def send_message():
    
    user_request = str(request.json['message'])

    chat_model = AzureChatOpenAI(
        openai_api_version = OPENAI_API_VERSION,
        azure_deployment = OPENAI_GPT_MODEL_DEP_NAME,
        temperature=0.1
    )

    chat_retriever = get_retriever()

    # Load or initialize the chat memory from the session
    qa_history = session.get('conversation_history', [])
    
    # You need to convert the list qa_history into conversation_history 
    # since conversation_history should follow the AI Prompt template 

    conversation_history = []
    if len(qa_history)>0:
        for it in qa_history:
            q = it[0]
            a = it[1]
            HumanMessage_v = HumanMessage(content = q)
            AI_v = AIMessage( content = a )
            conversation_history.append(HumanMessage_v)
            conversation_history.append(AI_v)
    else:
         conversation_history = [HumanMessage(content = "You are a good helper " ),
                                 AIMessage(content=" Thanks ")]
        
    bot_response = "This is a placeholder response based on the user's request."

    system_template = """
    You are an expert for credit card business, 
    You only answer questions related to to lending business and credit risk. 
    Ignore the personal identifiable information and answer generally. 
    ---------------
    {context}
    """

    human_template = """Previous conversation: {chat_history}
        Please provide an answer with less than 150 English words for the following new human question: {question}
        """
    
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ]
    

    print ('conversation_history', conversation_history)
    # Initialize the chain
    qa_prompt = ChatPromptTemplate.from_messages(messages)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        chain_type='stuff',
        retriever=chat_retriever,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )
        
    bot_response = qa({"question": user_request, "chat_history": conversation_history})['answer']

    # Append the conversation history to session state
    qu_an = [user_request, bot_response]
    if 'conversation_history' not in session:
        session['conversation_history'] = [] 

    session['conversation_history'].append(qu_an)
        
    session.modified = True  
    
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run()


