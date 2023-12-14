# 여기서부터 세줄은 로컬환경에서 돌릴 때에는(즉 웹사이트로 배포 안하고 그냥 터미널에서 돌릴때) 주석처리 해주셔야합니다. 
# 배포할때에는 주석처리하시면 안됩니다. 
# 주석처리 방법은 "Ctrl + "/"" 누르기
# ---------------------------------------------------
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ---------------------------------------------------

import os
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

st.set_page_config(page_title="자료 검증 챗봇", page_icon="👾")
st.title("자료 검증 챗봇")

# 로컬 환경에서 테스트해볼때
os.environ["OPENAI_API_KEY"] ="sk-blablabla"

# Streamlit 배포할때
# Streamlit 앱의 환경설정에서 꼭 OPENAI_API_KEY = "sk-blabalabla"를 추가해주세요!
# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# AI가 생성한 답변을 실시간으로 스트리밍해서 보여주는 함수 
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)

# 학습시킨 자료에서 답변을 생성하기 위해 retriever에서 불러온 내용을 보여주는 함수 (자료 검증)
class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


# 어떤 파일을 학습시키는지에 따라 코드를 바꿔주세요. ex) pdf, html, csv

# 첫번째 구현 방법: 웹사이트 url 학습시키기
# ---------------------------------------------------
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://dalpha.so/ko/howtouse?scrollTo=custom")
data = loader.load()
# ---------------------------------------------------


# 두번째 구현 방법: pdf 학습시키기
# 먼저 VSCode에서 만든 이 폴더 내에 pdf 파일을 업로드 해주셔야해요!
# 사용하고 싶으면 아래 부분의 코드 주석을 없애주세요
# ---------------------------------------------------
# from langchain.document_loaders import PyPDFLoader

# loader = PyPDFLoader("파일이름.pdf")
# pages = loader.load_and_split()

# data = []
# for content in pages:
#     data.append(content)
# ---------------------------------------------------


# 세번째 구현 방법: csv 학습시키기
# 먼저 VSCode에서 만든 이 폴더 내에 csv 파일을 업로드 해주셔야해요!
# 사용하고 싶으면 아래 부분의 코드 주석을 없애주세요
# ---------------------------------------------------
# from langchain.document_loaders.csv_loader import CSVLoader

# loader = CSVLoader(file_path='파일이름.csv')
# data = loader.load()
# ---------------------------------------------------

# 올린 파일 내용 쪼개기
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
all_splits = text_splitter.split_documents(data)

# 쪼갠 내용 vectorstore 데이터베이스에 업로드하기
vectordb = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# 데이터베이스에 업로드 한 내용을 불러올 수 있도록 셋업
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

# 대화 내용 기록하는 메모리 변수 셋업
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", temperature=0, streaming=True
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)

if len(msgs.messages) == 0 or st.sidebar.button("메세지 기록 삭제하기"):
    msgs.clear()
    msgs.add_ai_message("안녕하세요. 서비스 매뉴얼 챗봇입니다. 궁금한게 있으시면 물어봐주세요.")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Dalpha AI store는 어떻게 사용하나요?"):
    st.chat_message("user").write(user_query)

    # AI의 답변을 스트리밍하고 생성한 답변에 대한 증거 보여주기
    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])