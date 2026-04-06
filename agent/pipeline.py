import asyncio
from typing import AsyncGenerator
from .planner import plan_research
from .worker import execute_task
from .critic import critique_findings
from .synthesizer import synthesize_report


async def run_agent_pipeline(query: str) -> AsyncGenerator[dict, None]:
    """Orchestrate the full multi-agent research pipeline."""

    # ── Step 1: Planner ──────────────────────────────────────────────────────
    yield {"type": "step", "agent": "Planner", "status": "start",
           "message": "Analyzing query and creating research plan..."}

    try:
        tasks = await plan_research(query)
        if not tasks:
            yield {"type": "error", "message": "Planner returned no tasks."}
            return
        yield {"type": "step", "agent": "Planner", "status": "complete",
               "message": f"Created {len(tasks)} research sub-tasks", "data": tasks}
    except Exception as e:
        yield {"type": "error", "message": f"Planner error: {str(e)}"}
        return

    # ── Step 2: Workers (parallel) ────────────────────────────────────────────
    yield {"type": "step", "agent": "Workers", "status": "start",
           "message": f"Launching {len(tasks)} parallel research workers..."}

    for task in tasks:
        yield {"type": "task_start", "agent": "Worker",
               "task_id": task["id"],
               "message": f'Worker {task["id"]}: {task["task"][:80]}...'}

    try:
        findings = list(await asyncio.gather(*[execute_task(t) for t in tasks]))
        yield {"type": "step", "agent": "Workers", "status": "complete",
               "message": f"All {len(findings)} workers finished",
               "data": [{"task_id": f["task_id"], "task": f["task"],
                          "sources": f["sources"]} for f in findings]}
    except Exception as e:
        yield {"type": "error", "message": f"Worker error: {str(e)}"}
        return

    # ── Step 3: Critic ────────────────────────────────────────────────────────
    yield {"type": "step", "agent": "Critic", "status": "start",
           "message": "Evaluating research quality and identifying gaps..."}

    try:
        critique = await critique_findings(query, findings)
        score = critique.get("quality_score", "?")
        verdict = critique.get("verdict", "")
        yield {"type": "step", "agent": "Critic", "status": "complete",
               "message": f"Quality score: {score}/10 — {verdict}",
               "data": critique}
    except Exception:
        critique = {"quality_score": 7, "verdict": "sufficient",
                    "summary": "Proceeding with available findings."}
        yield {"type": "step", "agent": "Critic", "status": "complete",
               "message": "Review complete (fallback)"}

    # ── Step 4: Synthesizer (streaming) ──────────────────────────────────────
    yield {"type": "step", "agent": "Synthesizer", "status": "start",
           "message": "Writing comprehensive research report..."}

    all_sources = []
    for f in findings:
        all_sources.extend(f.get("sources", []))

    yield {"type": "report_start"}

    async for chunk in synthesize_report(query, findings, critique):
        yield {"type": "report_chunk", "content": chunk}

    yield {"type": "report_end", "sources": all_sources}
    yield {"type": "step", "agent": "Synthesizer", "status": "complete",
           "message": "Report complete!"}
