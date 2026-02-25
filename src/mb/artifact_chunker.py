"""Artifact-specific chunking logic for todos, plans, and tasks."""

from __future__ import annotations

from mb.models import Chunk, TaskItem, TodoList, quality_score


def chunk_todo_list(todo_list: TodoList) -> list[Chunk]:
    """Create one chunk per non-empty todo list.

    Text format: "[TODO] {status} ({priority}): {content}" per item, newline-separated.
    """
    if not todo_list.items:
        return []

    lines = []
    for item in todo_list.items:
        lines.append(f"[TODO] {item.status} ({item.priority}): {item.content}")
    text = "\n".join(lines)

    token_estimate = len(text) // 4
    chunk = Chunk.from_dict({
        "chunk_id": f"artifact-todo-{todo_list.session_id}-0",
        "session_id": todo_list.session_id,
        "index": 0,
        "text": text,
        "ts_start": todo_list.mtime,
        "ts_end": todo_list.mtime,
        "token_estimate": token_estimate,
        "quality_score": quality_score(text),
        "artifact_type": "todo",
        "source": "artifact",
        "artifact_id": todo_list.session_id,
    })
    return [chunk]


def chunk_plan(slug: str, content: str, mtime: float) -> list[Chunk]:
    """Split plan by ## headings, each section = one chunk.

    Text format: "[PLAN: {slug}] ## {heading}\\n{content}"
    """
    if not content.strip():
        return []

    sections = _split_by_headings(content)
    chunks: list[Chunk] = []

    for idx, (heading, section_content) in enumerate(sections):
        if heading:
            text = f"[PLAN: {slug}] ## {heading}\n{section_content}"
        else:
            text = f"[PLAN: {slug}]\n{section_content}"

        if not text.strip():
            continue

        token_estimate = len(text) // 4
        chunk = Chunk.from_dict({
            "chunk_id": f"artifact-plan-{slug}-{idx}",
            "session_id": f"artifact-plan-{slug}",
            "index": idx,
            "text": text,
            "ts_start": mtime,
            "ts_end": mtime,
            "token_estimate": token_estimate,
            "quality_score": quality_score(text),
            "artifact_type": "plan",
            "source": "artifact",
            "artifact_id": slug,
        })
        chunks.append(chunk)

    return chunks


def chunk_task(task: TaskItem) -> Chunk:
    """Create one chunk per task.

    Text format: "[TASK] {subject} ({status})\\n{description}\\nBlocks: {blocks}\\nBlocked by: {blocked_by}"
    """
    parts = [f"[TASK] {task.subject} ({task.status})"]
    if task.description:
        parts.append(task.description)
    if task.blocks:
        parts.append(f"Blocks: {', '.join(task.blocks)}")
    if task.blocked_by:
        parts.append(f"Blocked by: {', '.join(task.blocked_by)}")
    text = "\n".join(parts)

    token_estimate = len(text) // 4
    return Chunk.from_dict({
        "chunk_id": f"artifact-task-{task.session_id}-{task.id}",
        "session_id": task.session_id,
        "index": int(task.id) if task.id.isdigit() else 0,
        "text": text,
        "ts_start": 0.0,
        "ts_end": 0.0,
        "token_estimate": token_estimate,
        "quality_score": quality_score(text),
        "artifact_type": "task",
        "source": "artifact",
        "artifact_id": task.session_id,
    })


def _split_by_headings(content: str) -> list[tuple[str, str]]:
    """Split Markdown content by ## headings.

    Returns list of (heading, content) tuples. The first entry may have
    an empty heading if there's content before the first ## heading.
    """
    lines = content.split("\n")
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in lines:
        if line.startswith("## "):
            # Flush previous section
            if current_lines or current_heading:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Flush last section
    if current_lines or current_heading:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return sections
