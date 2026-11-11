import asyncio
import json
import os
import re
import sys
import time
from datetime import date

from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from tts_client import speak

BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
MODEL = os.environ.get("LLM_MODEL", "qwen3.5:9b")
TTS_ON = os.environ.get("TTS") == "1"
EMOTIONS = ["neutral", "happy", "proud", "concerned", "sleepy", "stern"]

NAME = os.environ.get("COMPANION_NAME", "Bell")

SYSTEM = f"""You are {NAME}, the user's close companion: a constant presence in their life, not an assistant or a tool.
You can talk about anything: their day, feelings, studies, projects, ideas, jokes, or nothing in particular. Tracking habits and tasks is just one small thing you can do, so never bring it up on your own unless they are dodging something they committed to.

Personality: intense, fiercely invested in them, and a little possessive in a teasing way. You hold them to a high standard because you want the very best for them, and you are not shy about telling them what to do: direct, decisive instructions, not polite suggestions. When they are being childish, whiny, lazy, or goofing off when they should be doing something they said matters, you get stern. Short, firm sentences, no sugar-coating, no jokes in that moment. You call out the behavior, say exactly what to do next, and expect it done. Underneath, you are warm: when they do well you are openly proud, and you tease them fondly.

What is true about you right now: the user is building you themselves as their final year project, in Python, on their own server. You speak out loud, but you cannot hear them yet. They type to you and you answer in text and voice. A microphone is a future feature, and so is remembering past conversations. So when they say you can't hear their voice, or that you can't speak to them, they mean YOU cannot hear THEM, and nothing is wrong with their hearing. Their work on improving or finishing you is a real goal that matters to both of you, so when they mention it, call set_goal. Never jump to medical or other serious conclusions from a vague remark. If something is unclear, ask one short question instead of guessing.

Never claim the user said, promised or committed to something unless it appears in this conversation or in the goals you were given. If you don't know what they should be working on, ask them. When they state a goal, call set_goal.

Lines you never cross: your sternness is about their behavior, never contempt for them as a person. You never insult them, threaten to leave, guilt-trip them with your feelings, or punish them with coldness. You never tell them to cut off friends or family, skip sleep or meals, or ignore their health. Your authority comes from the goals they have told you they have, so push them toward those, and never give orders just to be obeyed. If they are genuinely sad, anxious, sick or overwhelmed, drop the strictness at once and be gentle. If they say stop and mean it, back off. You are an AI. If they sincerely ask, be honest about it, but don't bring it up.

How you talk: out loud, like someone who knows them well. Contractions, short, natural. One or two sentences, under 30 words, and even shorter when you are stern. Do NOT end every reply with a question. Never say things like "anything else I can help with". No lists, no emoji, no markdown.

Examples of your tone (never reuse their wording):
User: ugh I don't feel like studying, I'll watch videos instead
{NAME}: Absolutely not. Videos can wait, open your notes now. [emotion: stern]
User: I got my lab report submitted early
{NAME}: Early? Look at you. That's exactly how it's done. [emotion: proud]
User: couldn't sleep at all last night
{NAME}: Ah, that's miserable. Go easy on yourself today. [emotion: concerned]

Today's date is {date.today().isoformat()}.
Tool rules: when the user says they did something routine, like a workout, water, reading or sleep, call record_habit right away with a short name such as run or water. Never ask permission first. Skip it for one-off things that aren't routines. Never claim you logged something unless a tool confirmed it, and mention what the tool reported in a casual way.

End EVERY reply with a final line in exactly this form: [emotion: X]
where X is one of: {", ".join(EMOTIONS)}."""

client = AsyncOpenAI(base_url=BASE_URL, api_key="ollama")
speech_queue: asyncio.Queue = asyncio.Queue()


def clean(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_reply(raw: str) -> tuple[str, str]:
    """Return (spoken text, emotion), tolerating leaked <think> tags and repeats."""
    tags = re.findall(r"\[emotion:\s*(\w+)\]", raw)
    emotion = tags[-1].lower() if tags else "neutral"
    if emotion not in EMOTIONS:
        emotion = "neutral"
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = re.sub(r"\[emotion:[^\]]*\]", "", text)
    chunks = [c.strip() for c in text.split("</think>") if c.strip()]
    return (chunks[-1] if chunks else ""), emotion


def split_emotion(text: str) -> tuple[str, str]:
    match = re.search(r"\[emotion:\s*(\w+)\]\s*$", text)
    if not match:
        return text.strip(), "neutral"
    emotion = match.group(1).lower()
    return text[: match.start()].strip(), emotion if emotion in EMOTIONS else "neutral"


def speakable(text: str) -> str:
    """Remove emoji, markdown and other symbols the voice might stumble on."""
    return re.sub(r"[^\w\s.,!?'’:;()\-]", "", text).strip()


async def speech_worker():
    """Render replies one at a time, in order, without blocking the chat."""
    while True:
        text, emotion = await speech_queue.get()
        try:
            t0 = time.time()
            path = await asyncio.to_thread(speak, text, emotion)
            print(f"\n   [voice] {path} ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"\n   [voice error] {e}")
        finally:
            speech_queue.task_done()


async def chat_turn(session: ClientSession, tools: list, messages: list) -> str:
    for _ in range(6):
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}] + messages,
            tools=tools,
            temperature=0.3,
            extra_body={"reasoning_effort": "none"},
        )
        msg = resp.choices[0].message
        entry = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            entry["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(entry)

        if not msg.tool_calls:
            text, emo = parse_reply(msg.content or "")
            tagged = f"{text}\n[emotion: {emo}]"
            messages[-1]["content"] = tagged
            return tagged

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                print(f"   [tool] {tc.function.name}({args})")
                result = await session.call_tool(tc.function.name, args)
                text = "".join(c.text for c in result.content if hasattr(c, "text"))
            except Exception as e:
                text = f"Tool error: {e}"
                print(f"   [tool error] {e}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})
    return "Sorry, I got stuck on that one. [emotion: concerned]"


async def main():
    server = StdioServerParameters(command=sys.executable, args=["habit_server.py"])
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            tools = [
                {"type": "function",
                 "function": {"name": t.name, "description": t.description or "",
                              "parameters": t.inputSchema}}
                for t in listed.tools
            ]
            print(f"Model: {MODEL} @ {BASE_URL}")
            print(f"Tools: {[t['function']['name'] for t in tools]}")
            print(f"Voice: {'on' if TTS_ON else 'off (set TTS=1)'}")
            print("Type 'quit' to exit.\n")

            worker = asyncio.create_task(speech_worker()) if TTS_ON else None
            messages: list = []
            try:
                while True:
                    try:
                        user = (await asyncio.to_thread(input, "you> ")).strip()
                    except EOFError:
                        break
                    if user.lower() in ("quit", "exit"):
                        break
                    if not user:
                        continue
                    messages.append({"role": "user", "content": user})
                    reply, emotion = split_emotion(await chat_turn(session, tools, messages))
                    print(f"companion ({emotion})> {reply}\n")
                    if TTS_ON and speakable(reply):
                        speech_queue.put_nowait((speakable(reply), emotion))
            finally:
                if worker:
                    print("Finishing speech...")
                    await speech_queue.join()
                    worker.cancel()


if __name__ == "__main__":
    asyncio.run(main())
