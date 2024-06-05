import os.path

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from streamlit.runtime.uploaded_file_manager import UploadedFile

from components.common.file_util import load_file
from components.langchain.file_parser import parse_by_file_and_disk_embedding, parse_by_file
from components.langchain.init_llm import initialize_open_ai_llm
from components.langchain.video_processing import extract_audio_from_video, cut_audio_in_chunks, transcribe_chunks
from components.pages.meetinggpt.prompt import find_other_summary_prompt, find_first_summary_prompt, find_qna_prompt
from components.util.util import get_file_name_from_path_str, format_docs

st.set_page_config(
    page_title="MeetingGPT",
    page_icon="💼",
)


@st.cache_resource
def init_open_ai() -> ChatOpenAI:
    return initialize_open_ai_llm()


@st.cache_data
def extract_audio(upload_file: UploadedFile) -> str:
    return extract_audio_from_video(
        video_file=upload_file,
        video_dir='./.cache/video_files',
        audio_dir='./.cache/audio_files')


@st.cache_data
def split_audio(audio_file_path_str: str, chunks_dir_str: str):
    cut_audio_in_chunks(
        audio_file_path=audio_file_path_str,
        minutes_in_chunk=10,
        chunks_dir=chunks_dir_str)


@st.cache_data
def exec_transcribe(chunks_dir_str: str, dest_path_str: str):
    transcribe_chunks(
        chunks_dir=chunks_dir_str,
        dest_path=dest_path_str
    )


st.markdown("""
    # MeetingGPT
    
    Welcome to MeetingGPT, upload a video and I will give you a transcript,
    a summary and a chat bot to ask any questions about it.
    
    Get started by uploading a video file in sidebar
""")

with st.sidebar:
    video_file = st.file_uploader("Video", type=['mp4', 'avi', 'mkv', 'mov'])

if video_file:
    with st.status("Extract audio...") as status:
        audio_file_path = extract_audio(video_file)

        status.update(label="Split audio file...")
        audio_file_path_underscore = get_file_name_from_path_str(audio_file_path).replace('.', '_')

        transcript_file_path = f'./.cache/transcript/{audio_file_path_underscore}.txt'
        if not os.path.exists(transcript_file_path):
            chunks_dir = f'./.cache/chunks/{audio_file_path_underscore}'
            split_audio(audio_file_path, chunks_dir)

        status.update(label="Making Transcript...")
        if not os.path.exists(transcript_file_path):
            exec_transcribe(chunks_dir, transcript_file_path)

    transcript_tab, summary_tab, qna_tab = st.tabs(['Transcript', 'Summary', 'Q&A'])

    with transcript_tab:
        transcript_file_contents = load_file(transcript_file_path)
        st.write(transcript_file_contents)

    with summary_tab:
        start = st.button("Generate Summary")

        # refine
        if start:
            llm = init_open_ai()
            docs = parse_by_file(transcript_file_path)

            first_summary_chain = find_first_summary_prompt() | llm | StrOutputParser()
            summary = first_summary_chain.invoke({"text": docs[0].page_content})

            refine_chain = find_other_summary_prompt() | llm | StrOutputParser()

            # 두번째 ~ 마지막 document 를 이용 하여 llm 을 실행 하면서 가공
            with st.status("Summarizing...") as summary_status:
                for i, doc in enumerate(docs[1:]):
                    summary_status.update(label=f"Processing document {i + 1}/{len(docs) - 1}")
                    summary = refine_chain.invoke({
                        "existing_summary": summary,
                        "context": doc.page_content,
                    })

            st.write(summary)

    with qna_tab:
        retriever = parse_by_file_and_disk_embedding(transcript_file_path,
                                                     f'./.cache/meeting_embeddings/{audio_file_path_underscore}')

        message = st.text_input("Ask anything about your transcript")


        if message:
            # case1 stuff chain
            chain = ({
                         "context": retriever | RunnableLambda(format_docs),
                         "question": RunnablePassthrough(),
                     } | find_qna_prompt() | init_open_ai() | StrOutputParser())

            resp = chain.invoke(message)
            st.write(resp)
