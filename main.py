import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Body, Form
from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

embeddings = OpenAIEmbeddings()
vector_store = PineconeVectorStore.from_existing_index("recipes", embeddings)

app = FastAPI(
    title="ChefGPT. The best provider of Indian Recipes in the world",
    description="Give ChefGPT a couple of ingredients and it will give you recipes in return",
    servers=[
        {"url": "https://firms-touch-charles-collectibles.trycloudflare.com"}
    ]
)


class Document(BaseModel):
    page_content: str


# uvicorn main:app --reload
# cloudflared tunnel --url http://127.0.0.1:8000 - localhost 를 외부 ssl 서버로 redirect
# windows 에서는 변경 사항이 정상적으로 적용되지 않는 경우가 있다 이경우에는 python 을 강제 종료하고 다시 시작하면 정상적으로 작동됨
# chatgpt oauth client_id: client123, client_secret: secret123 auth_url: /auth token_url: /token
# https://firms-touch-charles-collectibles.trycloudflare.com/openapi.json
@app.get(
    "/recipes",
    summary="Returns a list of recipes.",
    description="Upon receiving an ingredient, this endpoint will return a list of recipes that contain that "
                "ingredient.",
    response_description="A Document object that contains the recipe and preparation instructions",
    response_model=list[Document],
    openapi_extra={
        "x-openai-isConsequential": False,
    },
)
def get_recipe(ingredient: str):
    docs = vector_store.similarity_search(ingredient)
    return docs


user_token_db = {
    'ABCDEF': 'nico'
}


# queryString 의 key=value 값의 key 를 바로 파라 메터로 받을 수 있다
# include_in_schema : openapi.doc 에 포함할 지 여부
@app.get("/auth", response_class=HTMLResponse, include_in_schema=False)
def auth(redirect_uri: str, state: str):
    return f"""
    <html>
        <head>
            <title>Nicolacus Maximus Log In</title>
        </head>
        <body>
            <h1>Log Into Nicolacus Maximus</h1>
            <a href="{redirect_uri}?code=ABCDEF&state={state}">Authorize Nicolacus Maximus GPT</a>
        </body>
    </html>
    """


@app.post("/token", include_in_schema=False)
def token(code=Form(...)):
    print(code)
    return {
        'access_token': user_token_db[code]
    }
