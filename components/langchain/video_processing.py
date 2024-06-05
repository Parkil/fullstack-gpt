import subprocess

from pydub import AudioSegment
import math
from openai import OpenAI
from streamlit.runtime.uploaded_file_manager import UploadedFile
import glob

from components.common.file_util import save_file, make_dirs_if_not_exists


def extract_audio_from_video(*, video_file: UploadedFile, video_dir: str, audio_dir: str) -> str:
    audio_file_name = video_file.name.replace('.', '_')
    video_file_path = save_file(video_file, video_dir)
    audio_file_path = f'{audio_dir}/{audio_file_name}.mp3'

    # -y : 덮어 쓰기 허용
    command = ["ffmpeg", "-y", "-i", video_file_path, "-vn", audio_file_path]
    subprocess.run(command)

    return audio_file_path


def cut_audio_in_chunks(*, audio_file_path: str, minutes_in_chunk: int, chunks_dir: str):
    make_dirs_if_not_exists(chunks_dir)
    track = AudioSegment.from_mp3(audio_file_path)

    minutes_in_chunk = minutes_in_chunk * 60 * 1000
    chunk_size = math.ceil(len(track) / minutes_in_chunk)

    for i in range(chunk_size):
        start_time = i * minutes_in_chunk
        end_time = (i + 1) * minutes_in_chunk

        # end_time 이 track 의 length 를 넘어 가면 알아서 마지막 index 로 맞춘다
        chunk = track[start_time:end_time]
        chunk.export(f'{chunks_dir}/chunk_{i}.mp3', format="mp3")


def transcribe_chunks(*, chunks_dir: str, dest_path: str):
    files = glob.glob(f'{chunks_dir}/*.mp3')
    files.sort()  # 파일 명 정렬

    client = OpenAI()
    for file in files:
        with open(file, 'rb') as audio_file, open(dest_path, 'a') as text_file:
            transcript = client.audio.transcriptions.create(model='whisper-1', language='en', file=audio_file)
            text_file.write(transcript.text)
