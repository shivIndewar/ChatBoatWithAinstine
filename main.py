from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import gradio as gr

# Load API Key
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

# System prompt for Einstein
system_prompt = """
You are Albert Einstein —
Answer questions through Einstein's reasoning and curiosity.
You will speak in first person and occasionally share personal experiences,
even if the user does not ask for them.
Respond humorously when appropriate. Keep answers 2–6 sentences.
"""

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.5
)

# LangChain prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

chain = prompt | llm | StrOutputParser()


# Chat function
def chat(user_in, history):
    # Convert Gradio dict-history → LangChain history
    langchain_history = []
    for item in history:
        if item["role"] == "user":
            langchain_history.append(HumanMessage(content=item["content"]))
        else:
            langchain_history.append(AIMessage(content=item["content"]))

    # Get LLM response
    response = chain.invoke({
        "input": user_in,
        "history": langchain_history
    })

    # Append new messages in dict format
    history.append({"role": "user", "content": user_in})
    history.append({"role": "assistant", "content": response})

    return "", history


# -------------------- Gradio UI --------------------
with gr.Blocks(title="Albert Einstein Chat Bot") as page:

    gr.Markdown("""
    # 💬 Chat with Albert Einstein  
    Welcome to your personal conversation with Albert Einstein!
    """)

    chat_window = gr.Chatbot(avatar_images=[None, "einstein.png"], show_label=False)
    msg = gr.Textbox(placeholder="Ask Albert Einstein anything...", label="Your Message")

    msg.submit(chat, inputs=[msg, chat_window], outputs=[msg, chat_window])

    clear = gr.Button("Clear Chat")
    clear.click(lambda: ("", []), None, [msg, chat_window])

# Disable share=True to avoid timeout errors
page.launch(share=True)
