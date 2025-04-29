from langgraph.graph import StateGraph
from models import AgentState
from tools import analyze_sentiment, detect_toxicity, classify_tweet, generate_reply
from logger import log_interaction

# Create the graph
graph = StateGraph(AgentState)

# === Nodes ===

def sentiment_node(state: AgentState) -> AgentState:
    state.sentiment = analyze_sentiment.invoke({"tweet": state.tweet})
    return state

def toxicity_node(state: AgentState) -> AgentState:
    state.toxicity = detect_toxicity.invoke({"tweet": state.tweet})
    return state

def classification_node(state: AgentState) -> AgentState:
    state.category = classify_tweet.invoke({"tweet": state.tweet})
    return state

def reply_node(state: AgentState) -> AgentState:
    if state.toxicity:
        state.reply = "🚫 This tweet is too toxic to dignify with a response."
    else:
        state.reply = generate_reply.invoke({
            "tweet": state.tweet,
            "category": state.category,
            "sentiment": state.sentiment
        })
    return state

def logging_node(state: AgentState) -> AgentState:
    log_interaction(
        tweet=state.tweet,
        sentiment=state.sentiment,
        toxicity=state.toxicity,
        category=state.category,
        reply=state.reply
    )
    return state

# === Build the graph ===

graph.add_node("sentiment_node", sentiment_node)
graph.add_node("toxicity_node", toxicity_node)
graph.add_node("classification_node", classification_node)
graph.add_node("reply_node", reply_node)
graph.add_node("logging_node", logging_node)

graph.set_entry_point("sentiment_node")
graph.add_edge("sentiment_node", "toxicity_node")
graph.add_edge("toxicity_node", "classification_node")
graph.add_edge("classification_node", "reply_node")
graph.add_edge("reply_node", "logging_node")
graph.set_finish_point("logging_node")

# Compile agent
agent = graph.compile()