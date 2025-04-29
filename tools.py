from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from config import OPENAI_API_KEY, OPENAI_MODEL

# ─── Initialize LLM ─────────────────────────────────────────────────────────────
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=OPENAI_MODEL, temperature=0.3)


# ─── 1. Input Schemas ────────────────────────────────────────────────────────────
class SentimentInput(BaseModel):
    tweet: str

class ToxicityInput(BaseModel):
    tweet: str

class ClassificationInput(BaseModel):
    tweet: str

class GenerateReplyInput(BaseModel):
    tweet: str
    category: str
    sentiment: str


# ─── 2. Helper Functions ─────────────────────────────────────────────────────────
def _analyze_sentiment(tweet: str) -> str:
    prompt = f"Classify the sentiment of this tweet as Positive, Neutral, or Negative:\nTweet: {tweet}"
    return llm.invoke(prompt).content.strip()

def _detect_toxicity(tweet: str) -> bool:
    prompt = f"Is the following tweet toxic (hateful, violent, discriminatory)? Answer only 'True' or 'False'.\nTweet: {tweet}"
    return llm.invoke(prompt).content.strip().lower() == "true"

def _classify_tweet(tweet: str) -> str:
    prompt = f"Classify the tweet into: sarcastic, logical, funny, hateful, or fake.\nTweet: {tweet}"
    return llm.invoke(prompt).content.strip().lower()

def _generate_reply(tweet: str, category: str, sentiment: str) -> str:
    tone = (
        "savage and witty"
        if category in ["sarcastic", "funny", "fake", "hateful"] and sentiment != "Positive"
        else "respectful and clever"
    )
    prompt = (
        f"Write a {tone} reply to the following tweet based on its category '{category}' "
        f"and sentiment '{sentiment}':\nTweet: {tweet}"
    )
    return llm.invoke(prompt).content.strip()


# ─── 3. Wrap as Tools ────────────────────────────────────────────────────────────
analyze_sentiment = StructuredTool.from_function(
    name="analyze_sentiment",
    description="Analyze the sentiment (Positive, Neutral, Negative) of a tweet.",
    args_schema=SentimentInput,
    func=_analyze_sentiment,
)

detect_toxicity = StructuredTool.from_function(
    name="detect_toxicity",
    description="Detect if a tweet is toxic (hateful, violent, discriminatory).",
    args_schema=ToxicityInput,
    func=_detect_toxicity,
)

classify_tweet = StructuredTool.from_function(
    name="classify_tweet",
    description="Classify the tweet as sarcastic, logical, funny, hateful, or fake.",
    args_schema=ClassificationInput,
    func=_classify_tweet,
)

generate_reply = StructuredTool.from_function(
    name="generate_reply",
    description="Generate a witty or savage reply based on the tweet's classification and sentiment.",
    args_schema=GenerateReplyInput,
    func=_generate_reply,
)