from langchain_community.llms import GPT4All
from langchain.prompts import PromptTemplate
import time

start = time.time()

prompt = PromptTemplate.from_template("A {word} is a")

# model 에는 local model 명이 들어 간다 .bin 또는 .gguf 파일 형식이 들어 갈 수 있다
# HDD 와 SSD 에서 실행할 때 실핼 속도 차이가 많이 난다
# mistral-7b 를 실행할 때 처음 에는 7초 정도가 걸리 는데 다시 실행 하면 실행 시간이 엄청 나게 걸린다

# 사용자 정의 파일명 과 외부 라이브러리 명이 같으면 takes 1 positional argument but 2 were given (type=type_error) 같은 문제가 발생할 소지가 있다
# n_threads=8, device="nvidia" => 6.62738 sec
# n_threads=20, device="nvidia" => 23.64534 sec
# n_threads=4, device="nvidia" => 6.62524 sec
# n_threads=4, device="cpu" => 29.82329 sec
# 몇번 돌려본 결과 device 가 nvidia 일때 가장 성능이 좋았으며 HDD/SDD 차이도 있는 것으로 보인다
# 아직 까지는 추측이지만 소요시간의 대부분이 model을 loading 하는데 쓰이는 것으로 보인다
llm = GPT4All(model="../mistral-7b.gguf", n_threads=4, device="nvidia")

chain = prompt | llm

result = chain.invoke({
    "word": "tomato"
})

print(result)
end = time.time()
print(f"{end - start:.5f} sec")
