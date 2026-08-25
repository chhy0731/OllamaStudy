from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from yaspin import yaspin


llm = ChatOllama(
    base_url="http://1.238.159.68:11434",
    # model="qwen3-vl:30b-a3b-instruct",
    model="qwen3-vl:30b",
    temperature=0.9
)

# ai_msg = AIMessage("You have to explain simply")  #답변 컨셉 잡기

# sys_msg = SystemMessage(content=( "답변 전에 먼저 [THINKING]...[/THINKING] 블록에 추론 과정을 작성하고, 그 다음에 최종 답변을 출력해라.")) #답변 규칙 정하기 ex) 출력형식, 사고 스타일, 금지규칙, 역할 정체성


messages = [
    # ai_msg,
    # sys_msg,
    
]



while True:

    
    user = input("You: ")
    if user == "/bye":
        break

    messages.append(HumanMessage(content=user))

    print("Bot: ", end="", flush=True)

    response = ""
    reasoning_content = ''
    isThink = False
    is_pending = True
    first_chunk = False
    spinner_stopped = False
    new_chunk = ''

    spinner = yaspin(text=f"Pandding...", color="cyan")
    spinner.start() # spinner 시작 

    for chunk in llm.stream(messages, reasoning=True):
        if chunk.additional_kwargs.get('reasoning_content'): #이게 있으면 이 안에서 reasoning content 없으면 넘어가기 만일 if 문 없으면 none도 출력됨. 
            if not isThink:
                isThink = True
            reasoning_content += str(chunk.additional_kwargs.get('reasoning_content'))
            # lines = [line for line in reasoning_content.split('\n') if line][-5:]  # 마지막 줄 5개만 가져오기
            preview = reasoning_content[-80:].replace('\n', ' ').strip()
            spinner.text = f"\033[94mThinking...\033[0m {preview}"     
            # print(lines)


        
        # 답변 과정
        elif chunk.content and chunk.content != '':
            if isThink:
                isThink = False
            if is_pending:
                is_pending = False
                spinner.stop()
                spinner_stopped = True
                print("\n💬 Answer: ", end='', flush=True)
            cont = str(chunk.content)
            response += cont
            print(cont, end = '', flush=True)

        # 응답이 끝났을 경우
        elif getattr(chunk, "response_metadata", {}).get("done") == True: #response_metadata에서 done 키 값을 가져오기; done 키값은 스트리밍이 끝났다는 것을 의미 
            if not spinner_stopped: #is_pending 쪽 spinner가 안멈췄을 때만 spinner stop하기 안그러면 공백으로 다시 덮어씌워지면서 내용 없어짐. 
                spinner.stop() 
    
    print()
    print("======================")  # 줄바꿈
    print(response) 
    print()  # 줄바꿈
    messages.append(AIMessage(content=response))

