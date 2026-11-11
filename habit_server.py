import sqlite3
from datetime import date, timedelta
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("companion")
DB = "companion.db"

def db():
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS habits (id INTEGER PRIMARY KEY, name TEXT UNIQUE)")
    con.execute("CREATE TABLE IF NOT EXISTS logs (habit_id INTEGER, day TEXT, PRIMARY KEY (habit_id, day))")
    con.execute("CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY, title TEXT, done INTEGER DEFAULT 0, due TEXT)")
    return con

@mcp.tool()
def add_habit(name: str) -> str:
    """Start tracking a new habit."""
    with db() as con:
        con.execute("INSERT OR IGNORE INTO habits(name) VALUES (?)", (name.lower(),))
    return f"Now tracking '{name}'."

@mcp.tool()
def log_habit(name: str) -> str:
    """Mark a habit as done today."""
    today = date.today().isoformat()
    with db() as con:
        row = con.execute("SELECT id FROM habits WHERE name=?", (name.lower(),)).fetchone()
        if not row:
            return f"No habit named '{name}'. Add it first."
        con.execute("INSERT OR IGNORE INTO logs VALUES (?,?)", (row[0], today))
    return f"Logged '{name}' for {today}."

@mcp.tool()
def get_streak(name: str) -> str:
    """Current consecutive-day streak for a habit."""
    with db() as con:
        row = con.execute("SELECT id FROM habits WHERE name=?", (name.lower(),)).fetchone()
        if not row:
            return f"No habit named '{name}'."
        days = {r[0] for r in con.execute("SELECT day FROM logs WHERE habit_id=?", (row[0],))}
    d = date.today()
    if d.isoformat() not in days:
        d -= timedelta(days=1)
    streak = 0
    while d.isoformat() in days:
        streak += 1
        d -= timedelta(days=1)
    return f"{name}: {streak}-day streak."

@mcp.tool()
def add_task(title: str, due: str = "") -> str:
    """Add a task. due is optional, format YYYY-MM-DD."""
    with db() as con:
        con.execute("INSERT INTO tasks(title, due) VALUES (?,?)", (title, due or None))
    return f"Task added: {title}"

@mcp.tool()
def list_tasks() -> str:
    """List all unfinished tasks."""
    with db() as con:
        rows = con.execute("SELECT id, title, due FROM tasks WHERE done=0").fetchall()
    return "\n".join(f"[{i}] {t}" + (f" (due {d})" if d else "") for i, t, d in rows) or "No open tasks."

@mcp.tool()
def complete_task(task_id: int) -> str:
    """Mark a task as done by its id."""
    with db() as con:
        con.execute("UPDATE tasks SET done=1 WHERE id=?", (task_id,))
    return f"Task {task_id} completed."

@mcp.tool()
def list_habits() -> str:
    """List all habits being tracked, with whether each is done today."""
    today = date.today().isoformat()
    with db() as con:
        rows = con.execute(
            "SELECT h.name, EXISTS(SELECT 1 FROM logs l WHERE l.habit_id=h.id AND l.day=?) "
            "FROM habits h", (today,)).fetchall()
    return "\n".join(f"{n}: {'done today' if d else 'not done today'}" for n, d in rows) or "No habits yet."

@mcp.tool()
def record_habit(name: str) -> str:
    """Log that the user did a routine today (exercise, water, reading, sleep, meditation...). Call this immediately when they say they did one. It matches existing habits and creates the habit if it is new. Use a short name like 'run' or 'water'."""
    import difflib
    key = name.strip().lower()
    if not key:
        return "No habit name given."
    today = date.today().isoformat()
    with db() as con:
        names = [r[0] for r in con.execute("SELECT name FROM habits")]
        if key in names:
            match, created = key, False
        else:
            close = difflib.get_close_matches(key, names, n=1, cutoff=0.7)
            sub = [n for n in names if n in key or key in n]
            match = (close or sub or [None])[0]
            created = match is None
            if created:
                match = key
                con.execute("INSERT INTO habits(name) VALUES (?)", (key,))
        hid = con.execute("SELECT id FROM habits WHERE name=?", (match,)).fetchone()[0]
        newly = con.execute("INSERT OR IGNORE INTO logs VALUES (?,?)", (hid, today)).rowcount == 1
    if created:
        return f"Started a new habit '{match}' and logged it for today."
    if newly:
        return f"Logged '{match}' for today."
    return f"'{match}' was already logged today."

def gdb():
    con = db()
    con.execute("CREATE TABLE IF NOT EXISTS goals (id INTEGER PRIMARY KEY, text TEXT, deadline TEXT, done INTEGER DEFAULT 0)")
    return con

@mcp.tool()
def set_goal(text: str, deadline: str = "") -> str:
    """Save a goal or commitment the user clearly states they want to achieve, for example finishing a report or sleeping before 11pm. deadline is optional, format YYYY-MM-DD."""
    with gdb() as con:
        con.execute("INSERT INTO goals(text, deadline) VALUES (?,?)", (text.strip(), deadline or None))
    return f"Goal saved: {text.strip()}"

@mcp.tool()
def list_goals() -> str:
    """List the user's open goals with their ids."""
    with gdb() as con:
        rows = con.execute("SELECT id, text, deadline FROM goals WHERE done=0").fetchall()
    return "\n".join(f"[{i}] {t}" + (f" (due {d})" if d else "") for i, t, d in rows) or "No open goals."

@mcp.tool()
def complete_goal(goal_id: int) -> str:
    """Mark a goal as achieved by its id."""
    with gdb() as con:
        con.execute("UPDATE goals SET done=1 WHERE id=?", (goal_id,))
    return f"Goal {goal_id} marked done."

if __name__ == "__main__":
    mcp.run()
