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
            state["messages"] = []
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
            for _ in range(6):
                parser = ReplyParser()
                calls: dict = {}
                stream = await brain.client.chat.completions.create(
                    model=brain.MODEL,
                    messages=[{"role": "system", "content": build_web_system()}] + msgs,
                    tools=[t for t in state["tools"] if t["function"]["name"] != "set_goal"],
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
                spawn(extract_goals(text))
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
