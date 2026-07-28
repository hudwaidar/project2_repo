# pip install -qU langchain "langchain[openai]"
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

def summarize(text: str) -> str:
    """summerize the given text"""
    return f"{text}!"

agent = create_agent(
    model="openai:gpt-5.5",
    tools=[summarize],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "The new warehouse shipment includes fresh apples, winter jackets, and wireless mice.Bananas, heavy leather boots, and mechanical keyboards were also unloaded today.Workers stacked the crisp carrots right next to the gaming monitors and woolen scarves.Tomorrow, they expect oranges, high-end graphics cards, and a new line of athletic shoes"}]}
)
print(result["messages"][-1].content_blocks)