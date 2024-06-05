import os

from streamlit.runtime.uploaded_file_manager import UploadedFile


def save_file(file: UploadedFile, file_dir: str) -> str:
    file_content = file.read()
    file_path = f"{file_dir}/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path


def load_file(file_path: dir) -> str:
    with open(file_path, "r") as file:
        return file.read()


def make_dirs_if_not_exists(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
