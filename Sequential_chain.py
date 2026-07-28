# pip install -qU langchain "langchain[openai]"
from langchain.agents import create_agent
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

load_dotenv()

class ItemClassification(BaseModel):
    item: str = Field(description="The name of the item.")
    category: str = Field(description="The best category for the item.")

def summarize(text) -> str:
    """summerize the given text"""
    return f"{text}!"

categorizer = ChatOpenAI(model="gpt-4o-mini").with_structured_output(list[ItemClassification])


agent = create_agent(
    model="openai:gpt-5.5",
    tools=[summarize],
    system_prompt="You are a helpful assistant",
)

def format_for_agent(categorized_items) -> dict:
    return {"messages": [{"role": "user", "content": str(categorized_items)}]}

def count_words(agent_output) -> str:
    final_text = str(agent_output["messages"][-1].content)
    word_count = len(final_text.split())

    return f"Word Count: {word_count}\n\nFinal Text:\n{final_text}"

sequential_chain = categorizer | RunnableLambda(format_for_agent) | agent | RunnableLambda(count_words)

text = "The new warehouse shipment includes fresh apples, winter jackets, and wireless mice. Bananas, heavy leather boots, and mechanical keyboards were also unloaded today. Workers stacked the crisp carrots right next to the gaming monitors and woolen scarves. Tomorrow, they expect oranges, high-end graphics cards, and a new line of athletic shoes."



result = sequential_chain.invoke(text)

print(result)
