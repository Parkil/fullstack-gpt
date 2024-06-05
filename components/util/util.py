def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def get_file_name_from_path_str(path_str: str) -> str:
    split_arr = path_str.split('/')
    return split_arr[-1]
