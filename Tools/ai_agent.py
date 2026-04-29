from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.7,
    
)
model = ChatHuggingFace(llm=llm)
search_tool = DuckDuckGoSearchRun()
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm = model,
    tools = [search_tool],
    prompt = prompt
)

agent_executor = AgentExecutor(
    agent = agent,
    tools = [search_tool],
    verbose = True
)

response = agent_executor.invoke({"input":"3 ways to reach from blr to hyd"})
print(response)