from streamlit.runtime.uploaded_file_manager import UploadedFile


def save_file(file: UploadedFile, file_dir: str) -> str:
    file_content = file.read()
    file_path = f"{file_dir}/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path
