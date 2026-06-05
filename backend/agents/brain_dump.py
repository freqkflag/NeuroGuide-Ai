import os
from openai import OpenAI
from openai.agents import Agent, tool

# Initialize OpenAI client (API key from env)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@tool
def extract_tasks(text: str) -> list[dict]:
    """Parse free‑form user text into a list of task dictionaries.
    This mirrors the existing extraction logic used by the Responses API.
    The exact schema (title, category, priority, etc.) should match
    what the frontend expects.
    """
    # For now we simply call the OpenAI model with a system prompt.
    # In production this would be replaced by the same prompt/template
    # used in the legacy implementation.
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that extracts actionable tasks from user input and returns a JSON array of objects. Each object must contain at least 'title' and may include 'category', 'priority', and 'due_date'.",
            },
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    # The SDK automatically parses the JSON response.
    return response.choices[0].message.parsed

# Create a reusable agent instance.
BrainDumpAgent = Agent(
    name="BrainDumpAgent",
    system_prompt="You extract tasks from raw user input.",
    tools=[extract_tasks],
    client=client,
)
