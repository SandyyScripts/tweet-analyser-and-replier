from pydantic import BaseModel

class AgentState(BaseModel):
    tweet: str
    sentiment: str = None
    toxicity: bool = False
    category: str = None
    reply: str = None