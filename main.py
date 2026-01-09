from langchain_ollama import ChatOllama
import gradio as gr

model = ChatOllama(model="gemma3:12b", temperature=0)

def echo(message, history):
    response = model.invoke("대한민국 수도는 어디입니까?")
    return response

demo = gr.ChatInterface(fn=echo, examples=["hello", "hola", "merhaba"], title="Gemma Bot")
demo.launch()