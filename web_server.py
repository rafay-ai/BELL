import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).parent
os.chdir(HERE)  # habit_server.py uses a relative path for companion.db

import companion_local as brain  # noqa: E402
from tts_client import synthesize_stream  # noqa: E402

state: dict = {}


def trim(msgs: list, keep_turns: int = 20) -> None:
    """Keep only the last N user turns (cutting at turn starts keeps tool messages paired)."""
    starts = [i for i, m in enumerate(msgs) if m["role"] == "user"]
    if len(starts) > keep_turns:
        del msgs[: starts[-keep_turns]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    server = StdioServerParameters(
        command=sys.executable, args=[str(HERE / "habit_server.py")]
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            state["session"] = session
            state["tools"] = [
                {"type": "function",
                 "function": {"name": t.name, "description": t.description or "",
                              "parameters": t.inputSchema}}
                for t in listed.tools
            ]
            state["messages"] = load_turns()
            state["lock"] = asyncio.Lock()
            print(f"Tools: {[t['function']['name'] for t in state['tools']]}")
            yield


app = FastAPI(lifespan=lifespan)


class ChatIn(BaseModel):
    text: str


class SpeakIn(BaseModel):
    text: str
    emotion: str = "neutral"


@app.get("/", response_class=HTMLResponse)
async def index():
    return (HERE / "index.html").read_text(encoding="utf-8")


@app.post("/api/chat")
async def chat(body: ChatIn):
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "empty message")
    async with state["lock"]:
        msgs = state["messages"]
        n = len(msgs)
        msgs.append({"role": "user", "content": text})
        try:
            raw = await brain.chat_turn(state["session"], state["tools"], msgs)
        except Exception as e:
            del msgs[n:]  # drop the failed turn, including any half-finished tool calls
            raise HTTPException(502, f"LLM error: {e}")
        trim(msgs)
    reply, emotion = brain.split_emotion(raw)
    return {"reply": reply, "emotion": emotion}


@app.post("/api/speak")
async def speak(body: SpeakIn):
    text = brain.speakable(body.text)
    if not text:
        raise HTTPException(400, "nothing to say")
    it = synthesize_stream(text, body.emotion)
    try:
        first = await asyncio.to_thread(next, it, b"")  # surface connection errors early
    except Exception as e:
        raise HTTPException(502, f"TTS error: {e}")

    def body_iter():
        if first:
            yield first
        yield from it

    return StreamingResponse(
        body_iter(),
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.post("/api/reset")
async def reset():
    async with state["lock"]:
        state["messages"].clear()
        await asyncio.to_thread(clear_turns)
    return {"ok": True}


# ---------------- streaming chat ----------------
import json  # noqa: E402
import re  # noqa: E402
from datetime import date  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402


def goals_block() -> str:
    import sqlite3
    try:
        con = sqlite3.connect(str(HERE / "companion.db"))
        rows = con.execute("SELECT id, text, deadline FROM goals WHERE done=0").fetchall()
        con.close()
    except Exception:
        rows = []
    if not rows:
        return ("The user has not told you any goals yet. NEVER claim they promised, committed to, "
                "or said anything they did not say in this conversation. If you want to push them, "
                "ask what they are working on, and when they state a goal, call set_goal.\n\n")
    lines = "\n".join(f"- {t}" + (f" (due {d})" if d else "") for _, t, d in rows)
    return ("Goals the user has told you about (hold them to these and only these):\n" + lines +
            "\nNEVER claim they promised or said anything that is not in this list or this conversation. "
            "When they state a new goal, call set_goal.\n\n")


def _build_web_system_raw() -> str:
    """Same personality as the CLI, but the emotion tag comes FIRST so the voice knows it early."""
    s = brain.SYSTEM
    s = re.sub(r"(?m)^(\w+): (.*?) \[emotion: (\w+)\]\s*$", r"\1: [emotion: \3] \2", s)
    s = re.sub(r"Today's date is \d{4}-\d{2}-\d{2}",
               f"Today's date is {date.today().isoformat()}", s)
    cut = s.index("End EVERY reply")
    return s[:cut] + goals_block() + (
        "Format: begin EVERY reply with a first line exactly like [emotion: X], "
        f"where X is one of: {', '.join(brain.EMOTIONS)}. Then write what you say. "
        "Put the tag only at the start, never at the end."
    )


class ReplyParser:
    """Turns raw model tokens into events: emotion, delta (display text), sentence (to speak)."""

    def __init__(self):
        self.emotion = None
        self.tag_done = False
        self.head = ""       # held until the leading [emotion: X] tag is resolved
        self.pending = ""    # held back while a '[' might be the start of a tag
        self.sentbuf = ""
        self.shown = ""
        self.raw = ""
        self.blocked = False

    def feed(self, piece):
        events = []
        self.raw += piece
        if "think>" in self.raw:      # leaked reasoning tags: stop forwarding anything
            self.blocked = True
        if self.blocked:
            return events
        if not self.tag_done:
            self.head += piece
            m = re.match(r"\s*\[emotion:\s*(\w+)\]\s*", self.head)
            if m:
                self.tag_done = True
                emo = m.group(1).lower()
                self.emotion = emo if emo in brain.EMOTIONS else "neutral"
                events.append({"t": "emotion", "v": self.emotion})
                piece = self.head[m.end():]
            else:
                stripped = self.head.strip()
                if stripped and (not stripped.startswith("[") or len(stripped) > 40):
                    self.tag_done = True      # model didn't use a tag: treat it all as text
                    piece = self.head
                else:
                    return events
            self.head = ""
        self.pending += piece
        self._release(events, final=False)
        return events

    def _release(self, events, final):
        self.pending = re.sub(r"\[emotion:[^\]]*\]", "", self.pending)
        if final:
            out = re.sub(r"\[emotion:[^\]]*$", "", self.pending)
            self.pending = ""
        else:
            cut = self.pending.rfind("[")
            if cut == -1:
                out, self.pending = self.pending, ""
            else:
                out, self.pending = self.pending[:cut], self.pending[cut:]
        if not self.shown:
            out = out.lstrip()
        if out:
            self.shown += out
            self.sentbuf += out
            events.append({"t": "delta", "v": out})
            self._sentences(events, final=False)

    def _sentences(self, events, final):
        while True:
            cut = None
            for m in re.finditer(r"[.!?…]+[\"')\]]*\s+", self.sentbuf):
                if m.end() >= 15:         # merge very short sentences with the next one
                    cut = m.end()
                    break
            if cut is None:
                break
            s, self.sentbuf = self.sentbuf[:cut].strip(), self.sentbuf[cut:]
            if s:
                events.append({"t": "sentence", "v": s})
        if final and self.sentbuf.strip():
            events.append({"t": "sentence", "v": self.sentbuf.strip()})
            self.sentbuf = ""

    def finish(self):
        events = []
        if not self.tag_done and not self.blocked and self.head:
            self.tag_done = True
            self.pending += self.head
            self.head = ""
        self._release(events, final=True)
        self._sentences(events, final=True)
        return events


async def turn_events(text: str):
    async with state["lock"]:
        msgs = state["messages"]
        n = len(msgs)
        msgs.append({"role": "user", "content": text})
        completed = False
        try:
            just_did, recalled = await asyncio.gather(run_actions(text), recall(text))
            for note in just_did:
                yield {"t": "action", "v": note}
            for _ in range(6):
                parser = ReplyParser()
                calls: dict = {}
                stream = await brain.client.chat.completions.create(
                    model=brain.MODEL,
                    messages=[{"role": "system", "content": build_web_system(just_did, recalled)}] + msgs,
                    tools=[t for t in state["tools"] if t["function"]["name"] in READ_TOOLS],
                    temperature=0.3,
                    stream=True,
                    extra_body={"reasoning_effort": "none"},
                )
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        for ev in parser.feed(delta.content):
                            yield ev
                    for tc in (delta.tool_calls or []):
                        key = tc.index if tc.index is not None else len(calls)
                        c = calls.setdefault(key, {"id": "", "name": "", "args": ""})
                        if tc.id:
                            c["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                c["name"] = tc.function.name
                            if tc.function.arguments:
                                c["args"] += tc.function.arguments
                for ev in parser.finish():
                    yield ev

                emotion = parser.emotion or "neutral"
                shown = parser.shown.strip()
                entry = {"role": "assistant",
                         "content": f"[emotion: {emotion}]\n{shown}" if shown else ""}
                if calls:
                    ids = {k: (c["id"] or f"call_{k}") for k, c in calls.items()}
                    entry["tool_calls"] = [
                        {"id": ids[k], "type": "function",
                         "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                        for k, c in sorted(calls.items())
                    ]
                    msgs.append(entry)
                    for k, c in sorted(calls.items()):
                        yield {"t": "tool", "v": c["name"]}
                        print(f"[tool] {c['name']} {c['args']}", flush=True)
                        try:
                            result = await state["session"].call_tool(
                                c["name"], json.loads(c["args"] or "{}"))
                            out = "".join(x.text for x in result.content if hasattr(x, "text"))
                        except Exception as e:
                            out = f"Tool error: {e}"
                        print(f"[tool result] {out}", flush=True)
                        msgs.append({"role": "tool", "tool_call_id": ids[k], "content": out})
                    continue

                msgs.append(entry)
                completed = True
                trim(msgs)
                spawn(finish_turn(text, entry["content"], shown))
                yield {"t": "done"}
                return
            yield {"t": "error", "v": "I got stuck on that one."}
        finally:
            if not completed:
                del msgs[n:]   # drop a failed or abandoned turn so history stays consistent


@app.post("/api/chat_stream")
async def chat_stream(body: ChatIn):
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "empty message")

    async def gen():
        try:
            async for ev in turn_events(text):
                yield json.dumps(ev) + "\n"
        except Exception as e:
            yield json.dumps({"t": "error", "v": str(e)}) + "\n"

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


# ---------------- goals: saved by code, not by the persona ----------------
from datetime import timedelta  # noqa: E402
import habit_server as hs  # noqa: E402

bg: set = set()


def spawn(coro):
    t = asyncio.create_task(coro)
    bg.add(t)
    t.add_done_callback(bg.discard)


def week_table() -> str:
    d0 = date.today()
    return "; ".join(f"{(d0 + timedelta(days=i)).strftime('%A')} {(d0 + timedelta(days=i)).isoformat()}"
                     for i in range(8))


GOAL_TOOL = {"type": "function", "function": {
    "name": "set_goal",
    "description": "Save a goal or commitment the user clearly states they want to achieve. deadline is optional, YYYY-MM-DD.",
    "parameters": {"type": "object",
                   "properties": {"text": {"type": "string"}, "deadline": {"type": "string"}},
                   "required": ["text"]}}}


async def extract_goals(text: str):
    try:
        system = (f"Today and the coming days: {week_table()} (the first entry is today). "
                  "If the user's message states a goal, commitment or task they want to achieve, call set_goal "
                  "with a short text and a deadline as YYYY-MM-DD taken from the dates above when one is stated or implied. "
                  "If the message is not a goal, do not call any tool.")
        r = await brain.client.chat.completions.create(
            model=brain.MODEL, temperature=0.2, tools=[GOAL_TOOL],
            messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
            extra_body={"reasoning_effort": "none"},
        )
        for tc in (r.choices[0].message.tool_calls or []):
            if tc.function.name != "set_goal":
                continue
            args = json.loads(tc.function.arguments or "{}")
            goal = (args.get("text") or "").strip()
            if not goal:
                continue
            deadline = (args.get("deadline") or "").strip()

            def save():
                with hs.gdb() as con:
                    if con.execute("SELECT 1 FROM goals WHERE done=0 AND lower(text)=?",
                                   (goal.lower(),)).fetchone():
                        return "duplicate"
                    con.execute("INSERT INTO goals(text, deadline) VALUES (?,?)", (goal, deadline or None))
                return "saved"

            print(f"[goal] {await asyncio.to_thread(save)}: {goal} {deadline}", flush=True)
    except Exception as e:
        print(f"[goal error] {e}", flush=True)


def build_web_system() -> str:
    s = _build_web_system_raw()
    today = date.today()
    s = s.replace(f"Today's date is {today.isoformat()}",
                  f"Today's date is {today.strftime('%A')} {today.isoformat()}")
    s = s.replace("call set_goal", "acknowledge it")
    return s + ("\n\nGoals are saved automatically by a separate system. You have no tool for saving goals. "
                "Never say you are saving, logging, tracking or setting one; at most say you noted it.")


# ---------------- actions: all changes are made by code, not by the persona ----------------
READ_TOOLS = {"get_streak"}


def spawn(coro):
    t = asyncio.create_task(coro)
    bg.add(t)
    t.add_done_callback(bg.discard)
    return t


def _tool(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


_STR = {"type": "string"}
_INT = {"type": "integer"}
ACTION_TOOLS = [
    _tool("set_goal", "Save a goal or commitment the user states. deadline is YYYY-MM-DD.",
          {"text": _STR, "deadline": _STR}, ["text"]),
    _tool("complete_goal", "Mark a listed goal as achieved, by id.", {"goal_id": _INT}, ["goal_id"]),
    _tool("delete_goal", "Remove a listed goal. ONLY when the user explicitly asks to delete, remove or cancel it.",
          {"goal_id": _INT}, ["goal_id"]),
    _tool("record_habit", "Log that the user did a routine today (exercise, water, reading, sleep).",
          {"name": _STR}, ["name"]),
    _tool("add_task", "Add a small to-do the user asks you to remember. due is YYYY-MM-DD.",
          {"title": _STR, "due": _STR}, ["title"]),
    _tool("complete_task", "Mark a listed task as done, by id.", {"task_id": _INT}, ["task_id"]),
    _tool("note_feedback", "Save how the user wants you to behave or treat them, as a short instruction.",
          {"text": _STR}, ["text"]),
]


def snapshot():
    with hs.gdb() as con:
        con.execute("CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY, text TEXT, active INTEGER DEFAULT 1)")
        goals = con.execute("SELECT id, text, deadline FROM goals WHERE done=0").fetchall()
        tasks = con.execute("SELECT id, title, due FROM tasks WHERE done=0").fetchall()
        habits = [r[0] for r in con.execute("SELECT name FROM habits")]
        done_today = {r[0] for r in con.execute(
            "SELECT h.name FROM habits h JOIN logs l ON l.habit_id=h.id WHERE l.day=?",
            (date.today().isoformat(),))}
        fb = [r[0] for r in con.execute("SELECT text FROM feedback WHERE active=1 ORDER BY id DESC LIMIT 15")]
    return goals, tasks, habits, done_today, fb


def apply_action(name: str, a: dict):
    """Run one change. Returns a short note for the page, or None if nothing happened."""
    if name == "record_habit":
        return hs.record_habit(str(a.get("name", "")))
    if name == "add_task":
        title = str(a.get("title", "")).strip()
        return hs.add_task(title, str(a.get("due", "") or "")) if title else None
    with hs.gdb() as con:
        con.execute("CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY, text TEXT, active INTEGER DEFAULT 1)")
        if name == "set_goal":
            goal = str(a.get("text", "")).strip()
            if not goal:
                return None
            if con.execute("SELECT 1 FROM goals WHERE done=0 AND lower(text)=?", (goal.lower(),)).fetchone():
                return None
            deadline = str(a.get("deadline", "") or "").strip()
            con.execute("INSERT INTO goals(text, deadline) VALUES (?,?)", (goal, deadline or None))
            return f"Goal saved: {goal}" + (f" (due {deadline})" if deadline else "")
        if name in ("complete_goal", "delete_goal"):
            row = con.execute("SELECT text FROM goals WHERE id=? AND done=0", (int(a.get("goal_id")),)).fetchone()
            if not row:
                return None
            done = 1 if name == "complete_goal" else 2     # 2 = removed (restorable)
            con.execute("UPDATE goals SET done=? WHERE id=?", (done, int(a.get("goal_id"))))
            return ("Goal done: " if done == 1 else "Goal removed: ") + row[0]
        if name == "complete_task":
            row = con.execute("SELECT title FROM tasks WHERE id=? AND done=0", (int(a.get("task_id")),)).fetchone()
            if not row:
                return None
            con.execute("UPDATE tasks SET done=1 WHERE id=?", (int(a.get("task_id")),))
            return "Task done: " + row[0]
        if name == "note_feedback":
            fb = str(a.get("text", "")).strip()
            if not fb:
                return None
            con.execute("INSERT INTO feedback(text) VALUES (?)", (fb,))
            return "Noted: " + fb
    return None


async def run_actions(text: str) -> list:
    notes = []
    try:
        goals, tasks, habits, _, _ = await asyncio.to_thread(snapshot)
        fmt = lambda rows: "\n".join(f"[{i}] {t}" + (f" (due {d})" if d else "") for i, t, d in rows) or "none"
        system = (
            f"Today and the coming days: {week_table()} (the first entry is today).\n"
            f"Open goals:\n{fmt(goals)}\nOpen tasks:\n{fmt(tasks)}\n"
            f"Tracked habits: {', '.join(habits) or 'none'}\n\n"
            "Decide whether the user's latest message asks for any change. Call one tool per change; "
            "call nothing if none applies.\n"
            "- set_goal: they state a goal or commitment (deadline as YYYY-MM-DD from the dates above).\n"
            "- complete_goal / complete_task: they say they finished a listed goal or task (use its id).\n"
            "- delete_goal: ONLY when they explicitly ask to remove, delete or cancel a listed goal.\n"
            "- record_habit: they say they did a routine today. Short name; reuse an existing habit name if it fits.\n"
            "- add_task: they ask you to remember a small to-do.\n"
            "- note_feedback: they tell you how to behave or treat them (tone, strictness, what to stop doing).\n"
            "Never invent changes, and never delete anything they did not explicitly ask to delete.")
        r = await brain.client.chat.completions.create(
            model=brain.MODEL, temperature=0.2, tools=ACTION_TOOLS,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
            extra_body={"reasoning_effort": "none"})
        for tc in (r.choices[0].message.tool_calls or []):
            try:
                note = await asyncio.to_thread(
                    apply_action, tc.function.name, json.loads(tc.function.arguments or "{}"))
            except Exception as e:
                print(f"[action error] {tc.function.name}: {e}", flush=True)
                continue
            if note:
                print(f"[action] {note}", flush=True)
                notes.append(note)
    except Exception as e:
        print(f"[actions error] {e}", flush=True)
    return notes


def build_web_system() -> str:
    s = _build_web_system_raw()
    today = date.today()
    s = s.replace(f"Today's date is {today.isoformat()}",
                  f"Today's date is {today.strftime('%A')} {today.isoformat()}")
    s = s.replace("call set_goal", "acknowledge it")
    s = re.sub(r"Tool rules:[^\n]*\n*", "", s)
    goals, tasks, habits, done_today, fb = snapshot()
    extra = ("\n\nYou cannot change goals, tasks, habits or settings yourself: a separate system does it "
             "automatically and the screen shows a small note when it happens. Never say you saved, logged, "
             "removed, finished or changed something as if you did it yourself. You may use the read-only tools to look things up.")
    if habits:
        extra += "\nHabits today: " + ", ".join(f"{h} ({'done' if h in done_today else 'not yet'})" for h in habits)
    if tasks:
        extra += "\nOpen tasks: " + "; ".join(t for _, t, _ in tasks)
    if fb:
        extra += ("\nHow they have told you to treat them (follow it, but it never overrides the lines you never cross):\n"
                  + "\n".join(f"- {x}" for x in fb))
    return s + extra


@app.get("/api/state")
async def api_state():
    goals, tasks, habits, done_today, fb = await asyncio.to_thread(snapshot)
    return {"goals": goals, "tasks": tasks, "habits": habits,
            "done_today": sorted(done_today), "feedback": fb}


_prev_build = build_web_system


def build_web_system(just_did=None) -> str:
    s = _prev_build()
    if just_did:
        s += ("\n\nBecause of their latest message, the system just did this, and it is ALREADY DONE: "
              + "; ".join(just_did)
              + ". React to it in character, but never argue as if it had not happened.")
    return s


# ---------------- memory ----------------
import sqlite3  # noqa: E402
import numpy as np  # noqa: E402

EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3:567m")
MIN_SIM = float(os.environ.get("MEM_MIN_SIM", "0.45"))
DUP_SIM = 0.88


def mdb():
    con = sqlite3.connect(str(HERE / "companion.db"))
    con.execute("CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, text TEXT NOT NULL, "
                "day TEXT, emb BLOB, active INTEGER DEFAULT 1)")
    con.execute("CREATE TABLE IF NOT EXISTS turns (id INTEGER PRIMARY KEY, role TEXT, content TEXT)")
    return con


async def embed(texts):
    r = await brain.client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in r.data]


def unit(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def load_turns(n=12):
    try:
        con = mdb()
        rows = con.execute("SELECT role, content FROM turns ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        con.close()
    except Exception:
        return []
    msgs = [{"role": r, "content": c} for r, c in reversed(rows)]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


def save_turns(user_text, tagged_reply):
    con = mdb()
    con.execute("INSERT INTO turns(role, content) VALUES ('user', ?)", (user_text,))
    con.execute("INSERT INTO turns(role, content) VALUES ('assistant', ?)", (tagged_reply,))
    con.execute("DELETE FROM turns WHERE id NOT IN (SELECT id FROM turns ORDER BY id DESC LIMIT 200)")
    con.commit()
    con.close()


def clear_turns():
    con = mdb()
    con.execute("DELETE FROM turns")
    con.commit()
    con.close()


async def recall(text, k=5):
    """Memories relevant to this message, as (id, text, day) tuples."""
    try:
        q = unit((await embed([text]))[0])

        def search():
            con = mdb()
            rows = con.execute("SELECT id, text, day, emb FROM memories "
                               "WHERE active=1 AND emb IS NOT NULL").fetchall()
            recent = con.execute("SELECT id, text, day FROM memories WHERE active=1 "
                                 "ORDER BY id DESC LIMIT 2").fetchall()
            con.close()
            scored = sorted(((float(q @ np.frombuffer(b, dtype=np.float32)), i, t, d)
                             for i, t, d, b in rows), reverse=True)
            for sc, i, t, d in scored[:3]:
                print(f"[recall] {sc:.2f} {'(used)' if sc >= MIN_SIM else '(skipped)'} {t}", flush=True)
            hits = [(i, t, d) for sc, i, t, d in scored[:k] if sc >= MIN_SIM]
            return hits or recent   # no relevant hit (e.g. just "hey"): fall back to the latest two

        return await asyncio.to_thread(search)
    except Exception as e:
        print(f"[recall error] {e}", flush=True)
        return []


MEMORY_TOOL = _tool("remember", "Save one durable fact about the user worth remembering long-term.",
                    {"text": _STR}, ["text"])


async def extract_memories(user_text, reply):
    try:
        system = (
            f"Today is {date.today().strftime('%A')} {date.today().isoformat()}; coming days: {week_table()}.\n"
            "You maintain a long-term memory about the user. Read the user's message (and the companion's reply, "
            "for context only). Call remember once per distinct fact the USER revealed that is worth keeping "
            "long-term: facts about their life, the people in it, events and plans with dates, preferences, "
            "things they struggle with, how they feel about an ongoing situation, and how they want to be treated. "
            "Write each as a short third-person sentence with absolute dates (for example: Has a database exam on "
            "2026-11-20).\n"
            "Do NOT save: greetings or small talk; passing moods that will not matter later; anything only the "
            "companion said; goals, to-dos and habit logs (handled elsewhere); passwords, card or ID numbers; "
            "anything the user asks not to keep. If nothing qualifies, call nothing. Never invent details.")
        r = await brain.client.chat.completions.create(
            model=brain.MODEL, temperature=0.2, tools=[MEMORY_TOOL],
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": f"User said: {user_text}\nCompanion replied: {reply}"}],
            extra_body={"reasoning_effort": "none"})
        facts = []
        for tc in (r.choices[0].message.tool_calls or []):
            if tc.function.name == "remember":
                f = str(json.loads(tc.function.arguments or "{}").get("text", "")).strip()
                if f:
                    facts.append(f)
        if not facts:
            return
        vecs = [unit(v) for v in await embed(facts)]

        def save():
            con = mdb()
            existing = [np.frombuffer(b, dtype=np.float32) for (b,) in con.execute(
                "SELECT emb FROM memories WHERE active=1 AND emb IS NOT NULL")]
            for f, v in zip(facts, vecs):
                if any(float(v @ e) >= DUP_SIM for e in existing):
                    print(f"[memory] duplicate, skipped: {f}", flush=True)
                    continue
                con.execute("INSERT INTO memories(text, day, emb) VALUES (?,?,?)",
                            (f, date.today().isoformat(), v.tobytes()))
                existing.append(v)
                print(f"[memory] saved: {f}", flush=True)
            con.commit()
            con.close()

        await asyncio.to_thread(save)
    except Exception as e:
        print(f"[memory error] {e}", flush=True)


async def finish_turn(user_text, tagged_reply, shown):
    try:
        await asyncio.to_thread(save_turns, user_text, tagged_reply)
    except Exception as e:
        print(f"[turns error] {e}", flush=True)
    await extract_memories(user_text, shown)


_prev_build2 = build_web_system


def build_web_system(just_did=None, recalled=None) -> str:
    s = _prev_build2(just_did)
    if recalled:
        lines = "\n".join(f"- {('(' + d + ') ') if d else ''}{t}" for _, t, d in recalled)
        s += ("\n\nThings you remember about them from earlier conversations (dates show when you learned "
              "each). Use them naturally and only when relevant: don't recite them or announce that you "
              "remember. They are background facts, NOT promises or commitments. Never claim to remember "
              "anything that is not listed here; if they ask what you remember and it isn't here, say you "
              "don't.\n" + lines)
    return s


class MemIn(BaseModel):
    text: str


@app.get("/api/memories")
async def list_memories():
    def q():
        con = mdb()
        rows = con.execute("SELECT id, text, day FROM memories WHERE active=1 ORDER BY id DESC").fetchall()
        con.close()
        return rows
    return {"memories": [{"id": i, "text": t, "day": d} for i, t, d in await asyncio.to_thread(q)]}


@app.post("/api/memories")
async def add_memory(body: MemIn):
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "empty")
    vec = unit((await embed([text]))[0])

    def save():
        con = mdb()
        con.execute("INSERT INTO memories(text, day, emb) VALUES (?,?,?)",
                    (text, date.today().isoformat(), vec.tobytes()))
        con.commit()
        con.close()
    await asyncio.to_thread(save)
    return {"ok": True}


@app.delete("/api/memories/{mem_id}")
async def delete_memory(mem_id: int):
    def rm():
        con = mdb()
        con.execute("DELETE FROM memories WHERE id=?", (mem_id,))   # permanent on purpose
        con.commit()
        con.close()
    await asyncio.to_thread(rm)
    return {"ok": True}
