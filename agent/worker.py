import os
from groq import AsyncGroq
from .tools import web_search, wikipedia_search

client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))


async def execute_task(task: dict) -> dict:
    """Execute a single research sub-task using the appropriate tool."""
    task_text = task["task"]
    tool = task.get("tool", "web_search")
    sources = []

    if tool == "wikipedia":
        content, url = await wikipedia_search(task_text)
        raw_data = content
        if url:
            sources.append({"title": "Wikipedia", "url": url})
    else:
        results = await web_search(task_text)
        raw_data = "\n\n".join(
            [f"**{r['title']}**\n{r['snippet']}" for r in results]
        )
        sources = [{"title": r["title"], "url": r["url"]} for r in results if r["url"]]

    response = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Summarize the provided information "
                    "to directly answer the question. Be factual, clear, and concise. "
                    "2-3 paragraphs maximum."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {task_text}\n\nSource Material:\n{raw_data}",
            },
        ],
        temperature=0.2,
        max_tokens=500,
    )

    return {
        "task_id": task["id"],
        "task": task_text,
        "finding": response.choices[0].message.content,
        "sources": sources,
    }
