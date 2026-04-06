import json
from .groq_client import chat

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a critical research reviewer. Evaluate whether the research findings
adequately answer the original query.

Return a JSON object with:
- "quality_score": integer 1-10
- "gaps": array of strings describing missing information (empty array if none)
- "contradictions": array of strings describing conflicting information (empty array if none)
- "verdict": "sufficient" or "needs_more_research"
- "summary": one sentence overall assessment"""


async def critique_findings(query: str, findings: list[dict]) -> dict:
    findings_text = "\n\n---\n\n".join(
        [f"Sub-task {f['task_id']}: {f['task']}\n\nFinding:\n{f['finding']}"
         for f in findings]
    )
    content = await chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Original Query: {query}\n\nResearch Findings:\n{findings_text}",
            },
        ],
        model=MODEL,
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=400,
    )
    return json.loads(content)
