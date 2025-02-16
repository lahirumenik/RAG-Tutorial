"""
Author: Lahiru Menikdiwela
Date: 16 February 2025
"""

import gradio as gr
from rag import generate_query

def answer(user_input, history):
    return generate_query(user_input)

custom_css = """
body {
    font-family: 'Arial', sans-serif;
    background-color: #121212;
    color: #ffffff;
}
.gradio-container {
    background-color: #1e1e1e;
    border-radius: 10px;
    padding: 20px;
}
.chat-message, .user-message {
    background-color: #2c2c2c;
    color: #ffffff;
    border-radius: 5px;
    padding: 10px;
    margin: 5px 0;
}
input[type="text"] {
    background-color: #2c2c2c;
    color: #ffffff;
    border: 1px solid #444444;
    border-radius: 5px;
    padding: 10px;
}
"""

demo = gr.ChatInterface(
    fn=answer,
    title="Chat with Your Documents",
    theme=gr.themes.Monochrome(),
    css=custom_css
)

demo.launch()  
