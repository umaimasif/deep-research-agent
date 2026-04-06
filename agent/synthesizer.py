import os
from typing import AsyncGenerator
from groq import AsyncGroq

client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are an expert research writer. Write a comprehensive, well-structured report
based on the research findings provided.

Format the report using Markdown with these sections:
## Executive Summary
## Key Findings
(use sub-headers for each topic)
## Analysis & Insights
## Conclusion

Be thorough, insightful, and use clear language. Cite specific findings where relevant."""


async def synthesize_report(
    query: str, findings: list[dict], critique: dict
) -> AsyncGenerator[str, None]:
    """Stream a comprehensive final report."""
    findings_text = "\n\n".join(
        [f"### Research on: {f['task']}\n{f['finding']}" for f in findings]
    )
    quality_note = critique.get("summary", "")

    stream = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Research Query: {query}\n\n"
                    f"Research Findings:\n{findings_text}\n\n"
                    f"Quality Assessment: {quality_note}"
                ),
            },
        ],
        temperature=0.4,
        max_tokens=2000,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
