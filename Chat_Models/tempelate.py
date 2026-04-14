import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

st.title("Game Expert Assistant")

template = PromptTemplate.from_template(
    """You are a game expert assistant.

Game Type: {game_type}
Mode: {game_mode}
Game Name: {game_name}
User Question: {user_question}

Instructions:
- Give a clear and beginner-friendly answer.
- Include rules if relevant.
- Add 3 practical tips.
- Keep response short (max 150 words)."""
)

template.save("prompt1.json")
template2 = load_prompt("prompt1.json")

game_type = st.selectbox("Game Type", ["Indoor", "Outdoor"])
game_mode = st.selectbox("Mode", ["Physical", "Online"]) if game_type == "Indoor" else "N/A"
game_name = st.text_input("Game Name", "chess")
user_question = st.text_input("Your Question", "How can I improve?")

if st.button("Submit"):
    llm = GoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.5,
        timeout=None,
        max_retries=2,
        max_tokens=None,
        )
    chain = template2 | llm
    result = chain.invoke({
        'game_type':game_type,
        'game_mode':game_mode,
        'game_name':game_name,
        'user_question':user_question
    
    })
    st.write(result)