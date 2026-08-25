from langchain_community.chat_models import ChatOllama  # 또는 적절한 provider
from langchain_ollama import ChatOllama

llm = ChatOllama(
    base_url="",
    # model="qwen3-vl:30b-a3b-instruct",
    model="qwen3-vl:30b",
    temperature=0.9
)


while True:
    user = input("You: ")
    if user == "/bye":
        break
    print("Bot: ", end="", flush=True)

    
    for chunk in llm.stream("Solve: What is 15 * 23?"):
        # content에 추론 과정이 포함될 수 있음
        print(chunk.content, end="", flush=True)
        
        # additional_kwargs 확인
        if chunk.additional_kwargs:
            print(f"\nAdditional: {chunk.additional_kwargs}")