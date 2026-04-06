import json
import os
from groq import AsyncGroq


def get_client():
    return AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a research planner. Break the user's query into 3-5 focused sub-tasks.
Each sub-task should be a specific question answerable by searching the web or Wikipedia.

Return a JSON object with a "tasks" array. Each task has:
- "id": integer starting at 1
- "task": the specific research question (string)
- "tool": either "web_search" or "wikipedia"

Example:
{
  "tasks": [
    {"id": 1, "task": "What are transformer neural networks?", "tool": "wikipedia"},
    {"id": 2, "task": "Latest advances in transformer architecture 2024", "tool": "web_search"},
    {"id": 3, "task": "Real world applications of transformers in industry", "tool": "web_search"}
  ]
}"""


async def plan_research(query: str) -> list[dict]:
    response = await get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=600,
    )
    data = json.loads(response.choices[0].message.content)
    if isinstance(data, list):
        return data
    for val in data.values():
        if isinstance(val, list):
            return val
    return []
