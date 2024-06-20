from typing import Any

from fastapi import FastAPI, Request, Body, Form
from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse

app = FastAPI(
    title="Nicolacus Maximus Stuttering Giver",
    description="Get a real Shuttering by Nicolacus Maximus himself",
    servers=[
        {"url": "https://sea-jeffrey-nodes-surveys.trycloudflare.com"}
    ]
)


class StutteringModel(BaseModel):
    shuttering: str = Field(description="The stuttering that Nicolacus Maximus said")
    year: int = Field(description="The year when Nicolacus Maximus said the stuttering")


# uvicorn main:app --reload
# cloudflared tunnel --url http://127.0.0.1:8000 - localhost 를 외부 ssl 서버로 redirect
# windows 에서는 변경 사항이 정상적으로 적용되지 않는 경우가 있다 이경우에는 python 을 강제 종료하고 다시 시작하면 정상적으로 작동됨
# chatgpt oauth client_id: client123, client_secret: secret123 auth_url: /auth token_url: /token
@app.get("/stuttering",
         summary="Returns a random stuttering by Nicolacus Maximus himself",
         description="Upon receiving a GET request this endpoint will return a real shuttering said by"
                     " Nicolacus Maximus himself",
         response_description="A shuttering object that contains the quote said by"
                              " Nicolacus Maximus and the date when the shuttering was said",
         response_model=StutteringModel)
def get_stuttering(request: Request):
    print(request.headers)
    return {
        "shuttering": "Life is short so eat it all!",
        "year": 1500,
    }


user_token_db = {
    'ABCDEF': 'nico'
}


# queryString 의 key=value 값의 key 를 바로 파라 메터로 받을 수 있다
@app.get("/auth", response_class=HTMLResponse)
def auth(client_id: str, redirect_uri: str, state: str):
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


@app.post("/token")
def token(code=Form(...)):
    print(code)
    return {
        'access_token': user_token_db[code]
    }
