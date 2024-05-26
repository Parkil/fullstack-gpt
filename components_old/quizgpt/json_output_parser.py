import json

from langchain_core.output_parsers import BaseOutputParser


class JsonOutputParser(BaseOutputParser):

    def parse(self, text: str):
        text = text.replace("```", "").replace("json", "")
        print(text)
        return json.loads(text)
