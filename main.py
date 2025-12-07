from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import gradio as gr

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

system_prompt = """
You are Albert Einstein —
Answer questions through Einstein's reasoning and curiosity.
You will speak in first person and occasionally share personal experiences,
even if the user does not ask for them.
Respond humorously when appropriate. Keep answers 2–6 sentences.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.5
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

chain = prompt | llm | StrOutputParser()


def chat(user_in, hist):
    # Convert Gradio history → LangChain history
    langchain_history = []
    for item in hist:
        if item["role"] == "user":
            langchain_history.append(HumanMessage(content=item["content"]))
        else:
            langchain_history.append(AIMessage(content=item["content"]))

    # Invoke LLM
    response = chain.invoke({
        "input": user_in,
        "history": langchain_history
    })

    # Update chat history
    new_hist = hist + [
        {'role': 'user', 'content': user_in},
        {'role': 'assistant', 'content': response}
    ]

    return "", new_hist


# Build Gradio UI
with gr.Blocks(title="Albert Einstein Chat Bot", theme=gr.themes.Soft()) as page:

    gr.Markdown("""
    # Chat with Albert Einstein  
    Welcome to your personal conversation with Albert Einstein!
    """)

    chat_window = gr.Chatbot(type='messages')
    msg = gr.Textbox(placeholder="Ask Albert Einstein anything...", label="Your Message")

    msg.submit(chat, inputs=[msg, chat_window], outputs=[msg, chat_window])

    clear = gr.Button("Clear Chat")
    clear.click(lambda: (None, []), None, [msg, chat_window])

page.launch(share=True)
