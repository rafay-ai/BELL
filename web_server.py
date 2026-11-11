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


def build_web_system() -> str:
    """Same personality as the CLI, but the emotion tag comes FIRST so the voice knows it early."""
    s = brain.SYSTEM
    s = re.sub(r"(?m)^(\w+): (.*?) \[emotion: (\w+)\]\s*$", r"\1: [emotion: \3] \2", s)
    s = re.sub(r"Today's date is \d{4}-\d{2}-\d{2}",
               f"Today's date is {date.today().isoformat()}", s)
    cut = s.index("End EVERY reply")
    return s[:cut] + (
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
                    tools=state["tools"],
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
                        try:
                            result = await state["session"].call_tool(
                                c["name"], json.loads(c["args"] or "{}"))
                            out = "".join(x.text for x in result.content if hasattr(x, "text"))
                        except Exception as e:
                            out = f"Tool error: {e}"
                        msgs.append({"role": "tool", "tool_call_id": ids[k], "content": out})
                    continue

                msgs.append(entry)
                completed = True
                trim(msgs)
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
