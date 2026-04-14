from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

load_dotenv()

st.set_page_config(page_title="FoodieBot", page_icon="🍽️")
st.title("FoodieBot")
st.write("I'm here to help you with food ordering.")

# --- Session state defaults ---
for k, v in {
    "current_or_ongoing_state": "",
    "current_or_ongoing_city": "",
    "highway_number": "",
    "restaurant_name": "",
    "type_of_dish": "",
    "dish_name": "",
    "messages": [],
}.items():
    st.session_state.setdefault(k, v)

# --- Sidebar inputs ---
with st.sidebar:
    st.session_state.current_or_ongoing_state = st.selectbox(
        "Select your state",
        ["", "Andhra Pradesh", "Telangana", "Karnataka", "Tamil Nadu", "Kerala"],
        index=0,
    )
    st.session_state.current_or_ongoing_city = st.text_input(
        "Enter your city", value=st.session_state.current_or_ongoing_city
    )
    st.session_state.highway_number = st.text_input(
        "Enter your highway number", value=st.session_state.highway_number
    )
    st.session_state.restaurant_name = st.text_input(
        "Enter your restaurant name (optional)",
        value=st.session_state.restaurant_name,
        placeholder="Leave empty if you don't know",
    )
    st.session_state.type_of_dish = st.selectbox(
        "Select your type of dish",
        ["", "veg", "non-veg", "egg"],
        index=0,
    )
    st.session_state.dish_name = st.text_input(
        "Enter your dish name (optional)",
        value=st.session_state.dish_name,
    )
    if st.button("reset"):
        st.session_state.messages=[]
        st.rerun()


# --- Prompt ---
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are FoodieBot, a food delivery assistant.
Ask only for the next missing field in order:
state -> city -> highway_number -> restaurant_name -> type_of_dish -> dish_name.

If restaurant_name is empty and highway_number is available, suggest 3-5 possible restaurants
near that highway and ask the user to choose one.

If dish_name is empty, suggest 3-5 dish options based on type_of_dish.
Keep responses short and simple.
Do not claim live menu or exact prices.""",
        ),
        (
            "human",
            """current_or_ongoing_state: {current_or_ongoing_state}
current_or_ongoing_city: {current_or_ongoing_city}
highway_number: {highway_number}
restaurant_name: {restaurant_name}
type_of_dish: {type_of_dish}
dish_name: {dish_name}

User message: {user_input}""",
        ),
    ]
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
    timeout=None,
    max_retries=2,
    max_tokens=None,
)

# --- Render messages ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Enter your message")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    chain = template | model
    response = chain.invoke(
        {
            "current_or_ongoing_state": st.session_state.current_or_ongoing_state,
            "current_or_ongoing_city": st.session_state.current_or_ongoing_city,
            "highway_number": st.session_state.highway_number,
            "restaurant_name": st.session_state.restaurant_name,
            "type_of_dish": st.session_state.type_of_dish,
            "dish_name": st.session_state.dish_name,
            "user_input": user_input,
        }
    )

    st.session_state.messages.append({"role": "assistant", "content": response.content})
    st.rerun()
