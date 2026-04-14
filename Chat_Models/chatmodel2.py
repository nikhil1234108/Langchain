from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

st.header("Research institute")

game_selection = st.selectbox("Select type of the game", ["Indoor", "Outdoor"])

game_mode = "N/A"
if game_selection == "Indoor":
    game_mode = st.selectbox("Select the mode of game", ["Physical", "Online"])
    if game_mode == "Physical":
        game_name = st.selectbox("Select game name", ["ludo", "chess", "carroms"])
    else:
        game_name = st.selectbox("Select game name", ["mafia", "gta", "pubg"])
else:
    game_name = st.selectbox("Select game name", ["cricket", "football", "hockey"])

user_input = st.text_input("Clear your doubts here")

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

if st.button("Submit"):
    if not user_input.strip():
        st.warning("Enter your prompt")
    else:
        try:
            llm = GoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3,
                timeout=None,
                max_retries=2,
                max_tokens=None,
            )

            final_prompt = template.format(
                game_type=game_selection,
                game_mode=game_mode,
                game_name=game_name,
                user_question=user_input,
            )

            with st.spinner("Generating response..."):
                response = llm.invoke(final_prompt)
                st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")
