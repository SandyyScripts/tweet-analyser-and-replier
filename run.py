from agent import agent
from models import AgentState

if __name__ == "__main__":
    tweet = input("Paste a tweet: ")
    initial_state = AgentState(tweet=tweet)
    result = agent.invoke(initial_state)

    # result is an AddableValuesDict; treat it like a dict
    sentiment = result["sentiment"]
    toxicity  = result["toxicity"]
    category  = result["category"]
    reply     = result["reply"]

    print("\n==== Result ====")
    print("Sentiment:", sentiment)
    print("Toxicity:", toxicity)
    print("Category:", category)
    print("Savage/Respectful Reply:", reply)