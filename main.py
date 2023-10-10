from langchain.chains import LLMChain
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
import gradio as gr
import time

def get_custom_prompt():
    custom_prompt_template = """
    You are an AI Coding Assitant and your task is to solve coding problems and return code snippets based on given user's query. Below is the user's query.
    Query: {query}

    You just return the helpful code.
    Helpful Answer:
    """
    return custom_prompt_template

def set_custom_prompt():
    custom_prompt = get_custom_prompt()
    prompt_setting = PromptTemplate(template=custom_prompt,
        input_variables=['query'])
    return prompt_setting

def llm_model():
    code_model = CTransformers(
        model='codellama-7b-instruct.ggmlv3.Q4_0.bin',
        model_type="llama",
        max_new_tokens = 1096,
        temperature = 0.2,
        repetition_penalty = 1.13)
    return code_model

def code_bot():
    qa_prompt = set_custom_prompt()
    code_model = llm_model()
    mybot = LLMChain(prompt=qa_prompt,
             llm=code_model)
    return mybot

def bot_response(query):
    llmbot = code_bot()
    response = llmbot.run({'query':query})
    return response

with gr.Blocks(title='Code Llama Demo') as demo:
    # gr.HTML("Code Llama Demo")
    gr.Markdown("# Code Llama Demo")
    chatbot = gr.Chatbot([], elem_id="chatbot", height=700)
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        bot_message = bot_response(message)
        chat_history.append((message, bot_message))
        time.sleep(2)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()