import os
import sys
import json
import urllib.request
import urllib.error
from openai import OpenAI
from dotenv import load_dotenv

# Ensure Unicode prints correctly on Windows consoles
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

DATA_BASE_URL = os.environ["DATA_BASE_URL"]  # https://jhbelz.github.io/agentic-interaction-data/v1

client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"],  # https://api.groq.com/openai/v1
                api_key=os.environ["OPENAI_API_KEY"])     # dein Groq-Schlüssel

# ---------------------------------------------------------------------------
# Tool definition – passed to the model so it knows name, purpose & parameters
# ---------------------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_daily_appointments",
            "description": (
                "Retrieve all calendar appointments for a specific day. "
                "Returns a JSON object with the date and a list of events "
                "including id, start/end times, title, and location."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": (
                            "The date to look up in YYYY-MM-DD format, "
                            "e.g. '2026-10-29'."
                        ),
                    }
                },
                "required": ["date"],
            },
        },
    }
]


# ---------------------------------------------------------------------------
# Tool implementation – actually fetches the data from the remote API
# ---------------------------------------------------------------------------
def get_daily_appointments(date: str) -> str:
    """Fetch calendar appointments for *date* (YYYY-MM-DD) from DATA_BASE_URL."""
    url = f"{DATA_BASE_URL}/calendar/{date}.json"
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.read().decode()
    except urllib.error.HTTPError as exc:
        return json.dumps({"error": f"HTTP {exc.code}", "url": url})
    except urllib.error.URLError as exc:
        return json.dumps({"error": str(exc.reason), "url": url})


# Map of tool name → callable
tool_functions = {
    "get_daily_appointments": get_daily_appointments,
}

# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------
messages = [
    {"role": "user", "content": "What appointments do I have tomorrow, 2026-10-29?"}
]

print(">>> User:", messages[0]["content"])

# 1) First request – the model may decide to call a tool
response = client.chat.completions.create(
    model=os.environ["MODEL"],
    messages=messages,
    tools=tools,
)

assistant_msg = response.choices[0].message

# 2) If the model issued tool calls, execute them and send results back
if assistant_msg.tool_calls:
    # Append the assistant's message (with tool_calls) to the conversation
    messages.append(assistant_msg)

    for tool_call in assistant_msg.tool_calls:
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)
        print(f"    [tool] {fn_name}({fn_args})")

        result = tool_functions[fn_name](**fn_args)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
        )

    # 3) Second request – model formulates a final answer using the tool results
    response = client.chat.completions.create(
        model=os.environ["MODEL"],
        messages=messages,
        tools=tools,
    )
    assistant_msg = response.choices[0].message

print("<<< Assistant:", assistant_msg.content)
