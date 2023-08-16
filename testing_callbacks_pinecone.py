import streamlit as st

from langchain.document_loaders import RecursiveUrlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
from langsmith import Client

import pinecone
from langchain.vectorstores import Pinecone
import os
from langchain.vectorstores import Vectara

local = False


client = Client()

st.set_page_config(
    page_title="ChatLangChain",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_resource(ttl="1h")
def configure_retriever(vectorstore_choice='Pinecone'):
    if vectorstore_choice == 'Pinecone':
        if local:
            pinecone.init(
                api_key=os.getenv("PINECONE_API_FINN"),  # find at app.pinecone.io
                environment = os.getenv("PINECONE_ENV_FINN")  # next to api key in console
            )
            embeddings = OpenAIEmbeddings() 
        else:
            pinecone.init(
                api_key=st.secrets["PINECONE_API_FINN"], 
                environment = st.secrets["PINECONE_ENV_FINN"]
            )
            embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])

        index_name = "jorgen-phd-thesis"
        vectorstore = Pinecone.from_existing_index(index_name = index_name, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        return retriever
    
    elif vectorstore_choice == 'Vectara':
        if local:
            vectara_customer_id= ''
            vectara_corpus_id= ''
            vectara_api_key= ''
        else:
            vectara_customer_id=st.secrets["vectara_customer_id"]
            vectara_corpus_id=st.secrets["vectara_corpus_id"]
            vectara_api_key=st.secrets["vectara_api_key"]

        vectorstore = Vectara(
                vectara_customer_id=vectara_customer_id,
                vectara_corpus_id=vectara_corpus_id,
                vectara_api_key=vectara_api_key
        )
        retriever = vectorstore.as_retriever(n_sentence_context=200)

        return retriever

def reload_llm(model_choice="gpt-4", temperature=0, vectorstore_choice="Pinecone"):
    if local:
        llm = ChatOpenAI(temperature=temperature, streaming=True, model=model_choice, )
    else:
        llm = ChatOpenAI(temperature=temperature, streaming=True, model=model_choice, openai_api_key=st.secrets["openai_api_key"])
    message = SystemMessage(
        content=(
            "You are a helpful chatbot who is tasked with answering questions about the contents of the PhD thesis. "
            "Unless otherwise explicitly stated, it is probably fair to assume that questions are about the PhD thesis. "
            "If there is any ambiguity, you probably assume they are about that."
        )
    )
    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
    )

    tool = create_retriever_tool(
        configure_retriever(vectorstore_choice),
        "search_pdh_thesis",
        "Searches and returns text from PhD thesis. This tool should be used to answer questions about the PhD thesis.",
    )
    tools = [tool]

    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
    )
    memory = AgentTokenBufferMemory(llm=llm)
    print ("Reloaded LLM")
    return agent_executor, memory, llm


# Using "with" notation
with st.sidebar:
    with st.form('my_form'):
        model_choice = st.radio(
            "Model",
            ("gpt-4", "gpt-3.5-turbo-16k")
        )
        temperature = st.slider('Temperature', 0.0, 1.0, 0.0, 0.01)
        vectorstore_choice = st.radio(
            "Vectorstore",
            ("Pinecone", "Vectara")
        )
        submitted = st.form_submit_button('Reload LLM')
    if submitted: 
        reload_llm(model_choice=model_choice, temperature=temperature)
        print(model_choice, temperature)
    

"# Chat🦜🔗"


starter_message = "Ask me the PhD thesis!"
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [AIMessage(content=starter_message)]

agent_executor, memory, llm = reload_llm()

for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    memory.chat_memory.add_message(msg)


if prompt := st.chat_input(placeholder=starter_message):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        agent_executor, memory, llm = reload_llm(model_choice=model_choice, temperature=temperature, vectorstore_choice=vectorstore_choice)
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor(
            {"input": prompt, "history": st.session_state.messages},
            callbacks=[st_callback],
            include_run_info=True,
        )
        st.session_state.messages.append(AIMessage(content=response["output"]))
        st.write(response["output"])
        memory.save_context({"input": prompt}, response)
        st.session_state["messages"] = memory.buffer
        run_id = response["__run"].run_id
        print(llm)
