import os
from typing import Literal
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from pymilvus import MilvusClient, DataType
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

# from agents.import_node_model import import_node_model,import_node_prompt
# from agents.rules_node_model import rules_node_model,rules_node_prompt
# from agents.gneral_node import gneral_node,gneral_node_prompt
# from agents.main_router import main_router

from newAgents.chest_xray_agent import chest_xray_agent, chest_xray_prompt
from newAgents.doctor_referral_agent import doctor_referral_agent, doctor_referral_prompt
from newAgents.symptom_chat_agent import symptom_chat_agent, symptom_chat_prompt
from newAgents.lab_report_agent import lab_report_agent, lab_report_prompt
from newAgents.main_router import main_router
from newAgents.general_chat_agent import general_chat_agent, general_chat_prompt


from tools.Import_VectorDB_Searcher import Import_VectorDB_Searcher
from tools.Regulation_VectorDB_Searcher import Regulation_VectorDB_Searcher

tools=[Import_VectorDB_Searcher,Regulation_VectorDB_Searcher]
tool_node = ToolNode(tools)

def add_messages(left: list, right: list):
    """Add-don't-overwrite."""
    return left + right

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    history: any
    

def should_continue(state: AgentState) -> Literal["__end__"]:
    # Since we don't have a 'tools' node anymore, we always end after the model call
    return "__end__"
    
# def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
#     messages = state['messages']
#     last_message = messages[-1]
#     if last_message.tool_calls:
#       return "tools"

#     return "__end__"


# def call_model_import(state: AgentState):
#   messages = state['messages']
#   if isinstance(messages[-1], HumanMessage):
#     response = import_node_model.invoke(import_node_prompt.format(question = messages,history=state['history']))
#   else:
#     response = import_node_model.invoke(messages)
#   return {"messages": [response]}

# def call_model_regulation(state: AgentState):
#   messages = state['messages']
#   if isinstance(messages[-1], HumanMessage):
#     response = rules_node_model.invoke(rules_node_prompt.format(question = messages,history=state['history']))
#   else:
#     response = rules_node_model.invoke(messages)
#   return {"messages": [response]}

# def call_model_general(state: AgentState):
#   messages = state['messages']
#   if isinstance(messages[-1], HumanMessage):
#     print(state['history'])
#     response = gneral_node.invoke(gneral_node_prompt.format(question = messages,history=state['history']))
#   else:
#     response = gneral_node.invoke(messages)
#   return {"messages": [response]}

def call_model_lab_report(state: AgentState):
    messages = state['messages']
    if isinstance(messages[-1], HumanMessage):
        response = lab_report_agent.invoke(lab_report_prompt.format(question=messages[-1].content, history=state['history']))
    else:
        response = lab_report_agent.invoke(messages)
    return {"messages": [response]}

def call_model_symptom_chat(state: AgentState):
    messages = state['messages']
    if isinstance(messages[-1], HumanMessage):
        response = symptom_chat_agent.invoke(symptom_chat_prompt.format(question=messages[-1].content, history=state['history']))
    else:
        response = symptom_chat_agent.invoke(messages)
    return {"messages": [response]}

def call_model_chest_xray(state: AgentState):
    messages = state['messages']
    if isinstance(messages[-1], HumanMessage):
        response = chest_xray_agent.invoke(chest_xray_prompt.format(question=messages[-1].content, history=state['history']))
    else:
        response = chest_xray_agent.invoke(messages)
    return {"messages": [response]}

def call_model_doctor_referral(state: AgentState):
    messages = state['messages']
    if isinstance(messages[-1], HumanMessage):
        response = doctor_referral_agent.invoke(doctor_referral_prompt.format(question=messages[-1].content, history=state['history']))
    else:
        response = doctor_referral_agent.invoke(messages)
    return {"messages": [response]}

def call_model_general_chat(state: AgentState):
    messages = state['messages']
    if isinstance(messages[-1], HumanMessage):
        response = general_chat_agent.invoke(general_chat_prompt.format(question=messages[-1].content, history=state['history']))
    else:
        response = general_chat_agent.invoke(messages)
    return {"messages": [response]}

def decide_the_path(state):
    decide = main_router.invoke(state["messages"])
    st = decide['datasource']
    print(st)
    return st


workflow = StateGraph(AgentState)

# workflow.add_node("tools", tool_node)
# workflow.add_node("gneral_node", call_model_general)
# workflow.add_node("import_node", call_model_import)
# workflow.add_node("rules_node", call_model_regulation)

# workflow.set_conditional_entry_point(
#     decide_the_path,
#     {
#         "pricing": "import_node",
#         "regulations": "rules_node",
#         "general": "gneral_node",
#     },
# )


# workflow.add_conditional_edges("gneral_node",should_continue)
# workflow.add_conditional_edges("import_node",should_continue)
# workflow.add_conditional_edges("rules_node",should_continue)

# workflow.add_edge('tools', 'rules_node')
# workflow.add_edge('tools', 'import_node')


# workflow.add_node("tools", tool_node)  # Remove tool node reference
workflow.add_node("lab_report", call_model_lab_report)
workflow.add_node("symptom_chat", call_model_symptom_chat)
workflow.add_node("chest_xray", call_model_chest_xray)
workflow.add_node("doctor_referral", call_model_doctor_referral)
workflow.add_node("general_chat", call_model_general_chat)

workflow.set_conditional_entry_point(
    decide_the_path,
    {
        "lab_report": "lab_report",
        "symptom_chat": "symptom_chat",
        "chest_xray": "chest_xray",
        "doctor_referral": "doctor_referral",
        "general_chat": "general_chat",

    },
)

workflow.add_conditional_edges("lab_report", should_continue)
workflow.add_conditional_edges("symptom_chat", should_continue)
workflow.add_conditional_edges("chest_xray", should_continue)
workflow.add_conditional_edges("doctor_referral", should_continue)
workflow.add_conditional_edges("general_chat", should_continue)



finalAgent = workflow.compile()



