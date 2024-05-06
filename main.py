from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

load_dotenv(verbose=True)  # .env 파일에 있는 설정 값을 불러 온다

# llm = OpenAI() text-davinci-003 (deprecated) 2024-01 에 폐기됨 아마 버전을 높이면 OpenAI 에서 다른 LLM Model 로
# 변경될 것으로 예상
chat = ChatOpenAI()  # gpt-3.5-turbo

b = chat.predict("How many planets are there?")
print(b)
