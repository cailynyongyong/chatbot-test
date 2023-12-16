# 여기서부터 세줄은 로컬환경에서 돌릴 때에는(즉 웹사이트로 배포 안하고 그냥 터미널에서 돌릴때) 주석처리 해주셔야합니다. 
# 배포할때에는 주석처리하시면 안됩니다. 
# 주석처리 방법은 "Ctrl + "/"" 누르기
# ---------------------------------------------------
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ---------------------------------------------------

# 필요한 모듈들 불러오기 
import os
import streamlit as st
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

st.set_page_config(page_title="DALL-E Chatbot", page_icon="🌠")
st.title("DALLE Chatbot")

# 로컬 환경에서 테스트해볼때
os.environ["OPENAI_API_KEY"] ="내 api key 넣기"

# Streamlit 배포하고 싶다면
# Streamlit 앱의 환경설정에서 꼭 OPENAI_API_KEY = "sk-blabalabla"를 추가해주세요!
# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# OpenAI LLM 셋업하기. temperature = 0.9는 더 창의적인 프롬프트를 생성하라는 뜻.
llm = OpenAI(temperature=0.9)
# LLM 프롬프트 만들어주기
# 유저의 인풋에 따라 DALL-E에 넣을 더 자세한 프롬프트를 
# 생성하라고 template 변수 안에 명시합니다.
prompt = PromptTemplate(
    input_variables=["image_desc"],
    template="Generate a detailed prompt to generate an image based on the following description: {image_desc}",
)
# LLM 체인을 위에서 셋업한 llm과 prompt를 사용해서 활성화시켜줍니다. 
chain = LLMChain(llm=llm, prompt=prompt)

# 대화 내용 기록하는 메모리 변수 셋업
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# 웹 UI 인터페이스 구성
if len(msgs.messages) == 0 or st.sidebar.button("메세지 기록 삭제하기"):
    msgs.clear()
    msgs.add_ai_message("안녕하세요. DALLe 이미지 생성 챗봇입니다. 원하는 사진을 생성하세요!")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Design a greeting card for Christmas"):
    st.chat_message("user").write(user_query)

    # 생성된 이미지 보여주기!
    with st.chat_message("assistant"):
        image_url = DallEAPIWrapper().run(chain.run(user_query))
        st.image(image_url)