import streamlit
from langchain_core.callbacks import BaseCallbackHandler

from components_old.streamlit_component import save_message


# st.empty() 로 호출할 경우 set_page_config() can only be called once per app page
# 에러가 발생 하지만 streamlit.empty() 로 처리할 경우 에는 오류가 발생 하지 않는다
# 소스 코드를 봤을때 에는 딱히 set_page_config 를 호출 하는 부분이 없었 는데
class ChatCallBackHandler(BaseCallbackHandler):
    __message = ""
    __message_box = None

    # args - 일반 적인 param ex) ('1',2,'3'...)
    # kwargs - keyword param ex) (a=1,b=2,.....)
    def on_llm_start(self, *args, **kwargs):
        self.__message = ""
        self.__message_box = streamlit.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.__message, 'ai')

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.__message += token
        self.__message_box.markdown(self.__message)
