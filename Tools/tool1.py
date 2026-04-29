from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()
@tool
def get_multiply(a:int, b:int) -> int:
    """Multiply two integers and return the product."""
    return a*b

def run_tool_demo() -> None:
    print(get_multiply.invoke({'a': 10, 'b': 20}))
    print(get_multiply.description)
    print(get_multiply.args)
    print(get_multiply.name)

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, max_retries=2)
    llm_with_tools = model.bind_tools([get_multiply])

    messages = [HumanMessage(content="Use get_multiply to multiply 10 and 20.")]
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        first_tool_call = response.tool_calls[0]
        tool_args = first_tool_call["args"]
        tool_output = get_multiply.invoke(tool_args)
        print(f"Tool output: {tool_output}")
        messages.append(
            ToolMessage(content=str(tool_output), tool_call_id=first_tool_call["id"])
        )
        final_response = llm_with_tools.invoke(messages)
        print(final_response.content)
    else:
        print(response.content)


if __name__ == "__main__":
    try:
        run_tool_demo()
    except Exception as exc:
        print(f"tool1.py execution failed: {exc}")

