from flask import Flask, render_template_string
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

class AgentState(TypedDict):
    input_text: str
    summary_result: str
    count_result: str

summarize_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    system_prompt="You are a helpful assistant. Summarize the provided text.",
)

count_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    system_prompt="You are a helpful assistant. Count how many times the word 'and' appears in the provided text.",
)

def summarize_node(state: AgentState):
    response = summarize_agent.invoke({
        "messages": [{"role": "user", "content": f"Summarize this text: {state['input_text']}"}]
    })
    return {"summary_result": response["messages"][-1].content}

def count_node(state: AgentState):
    response = count_agent.invoke({
        "messages": [{"role": "user", "content": f"Count the word 'and' in this text: {state['input_text']}"}]
    })
    return {"count_result": response["messages"][-1].content}

workflow = StateGraph(AgentState)
workflow.add_node("summarize", summarize_node)
workflow.add_node("count", count_node)
workflow.set_entry_point("summarize")
workflow.add_edge("summarize", "count")
workflow.add_edge("count", END)

graph_app = workflow.compile()

HTML_PAGE = """
<h3>Input:</h3> <p>{{ result.input_text }}</p>
<h3>Summary:</h3> <p>{{ result.summary_result }}</p>
<h3>Count:</h3> <p>{{ result.count_result }}</p>
"""

@app.route("/")
def home():
    my_sample_text = "The new warehouse shipment includes fresh apples, winter jackets, and wireless mice. Bananas, heavy leather boots, and mechanical keyboards were also unloaded today. Workers stacked the crisp carrots right next to the gaming monitors and woolen scarves. Tomorrow, they expect oranges, high-end graphics cards, and a new line of athletic shoes"
    
    result = graph_app.invoke({"input_text": my_sample_text})
    return render_template_string(HTML_PAGE, result=result)

if __name__ == "__main__":
    app.run(debug=True)