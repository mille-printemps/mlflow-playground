import argparse
import os

import mlflow
import mlflow.langchain
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from tools import SERVING_URL, classify_sentiment

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a sentiment classification tool. "
    "When a user shares text and asks about its sentiment, tone, or opinion, call the "
    "classify_sentiment tool rather than judging it yourself."
)


def build_agent():
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    return create_agent(llm, tools=[classify_sentiment], system_prompt=SYSTEM_PROMPT)


def run_query(agent, user_input: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    return result["messages"][-1].content


def main():
    parser = argparse.ArgumentParser(description="Chat with the sentiment agent.")
    parser.add_argument("query", nargs="*", help="One-shot query; omit for an interactive loop.")
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("sentiment-agent")
    mlflow.langchain.autolog()

    agent = build_agent()

    if args.query:
        print(run_query(agent, " ".join(args.query)))
        return

    print(
        f"Sentiment agent ready (model={OLLAMA_MODEL}, tool endpoint={SERVING_URL}). "
        "Ctrl+C to exit."
    )
    while True:
        try:
            user_input = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input.strip():
            continue
        print(f"Agent: {run_query(agent, user_input)}")


if __name__ == "__main__":
    main()
