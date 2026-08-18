import os
import sys
import io
import traceback
from typing import TypedDict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================================================
# 1. GEMINI API KEY
# =========================================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )


# =========================================================
# 2. LLM INITIALIZATION
# =========================================================

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key,
    temperature=0
)

llm = llm_flash


# =========================================================
# 3. STATE DEFINITION
# =========================================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# =========================================================
# 4. TOOLS
# =========================================================

@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return the standard output or error trace."""

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()

    sys.stdout = new_stdout

    try:
        local_scope = {}

        exec(
            clean_code,
            {},
            local_scope
        )

        result = new_stdout.getvalue()

    except Exception:
        result = (
            "Execution Error:\n"
            + traceback.format_exc()
        )

    finally:
        sys.stdout = old_stdout

    return (
        result.strip()
        if result.strip()
        else "Success (no terminal output)"
    )


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate specific test scenarios for a coding task."""

    prompt = (
        "You are a Senior QA Engineer. "
        "Generate 3 to 5 highly specific test scenarios "
        "for the following coding task:\n\n"
        f"{task_description}\n\n"
        "Include standard cases and edge cases. "
        "Return them as a numbered list."
    )

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(
                        item.get("text", "")
                    )

            elif isinstance(item, str):
                text_parts.append(item)

        return "".join(text_parts)

    return str(content)


# =========================================================
# 5. GRAPH NODES
# =========================================================

def task_input_node(state: CrewState):

    return state


def real_time_developer(state: CrewState):

    task = state["messages"][-1].content

    dev_prompt = (
        "Write a clean Python script to solve this task:\n\n"
        f"{task}\n\n"
        "Only return the Python code. "
        "Do not include explanations or markdown formatting."
    )

    response = llm_flash.invoke(dev_prompt)

    content = response.content

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":
                    text_parts.append(
                        item.get("text", "")
                    )

            elif isinstance(item, str):
                text_parts.append(item)

        code_str = "".join(text_parts)

    else:

        code_str = str(content)

    return {
        "code": code_str
    }


def real_time_tester(state: CrewState):

    task = state["messages"][-1].content

    # Generate test cases
    test_cases = generate_test_cases.invoke(task)

    cases_str = str(test_cases)

    # Execute generated code
    execution_result = run_python_code.invoke(
        {
            "code": state["code"]
        }
    )

    # Create report
    report = (
        "### EXECUTION OUTPUT:\n"
        f"{execution_result}\n\n"
        "### TEST SCENARIOS EVALUATED:\n"
        f"{cases_str}"
    )

    return {
        "report": report
    }


# =========================================================
# 6. GRAPH CONSTRUCTION
# =========================================================

rt_workflow = StateGraph(CrewState)

rt_workflow.add_node(
    "task_input",
    task_input_node
)

rt_workflow.add_node(
    "developer",
    real_time_developer
)

rt_workflow.add_node(
    "tester",
    real_time_tester
)


rt_workflow.add_edge(
    START,
    "task_input"
)

rt_workflow.add_edge(
    "task_input",
    "developer"
)

rt_workflow.add_edge(
    "developer",
    "tester"
)

rt_workflow.add_edge(
    "tester",
    END
)


rt_app = rt_workflow.compile()


# =========================================================
# 7. FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Real-Time Coding Agent",
    version="1.0.0"
)


# =========================================================
# 8. REQUEST MODEL
# =========================================================

class TaskRequest(BaseModel):
    task: str


# =========================================================
# 9. HOME ENDPOINT
# =========================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "message": "Real-Time Coding Agent is running."
    }


# =========================================================
# 10. HEALTH ENDPOINT
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# =========================================================
# 11. RUN AGENT
# =========================================================

@app.post("/run")
def run_agent(request: TaskRequest):

    initial_state: CrewState = {
        "messages": [
            HumanMessage(
                content=request.task
            )
        ],
        "next_step": None,
        "code": None,
        "report": None
    }

    try:

        result = rt_app.invoke(
            initial_state,
            config={
                "recursion_limit": 50
            }
        )

        return {
            "task": request.task,
            "generated_code": result.get(
                "code",
                ""
            ),
            "report": result.get(
                "report",
                ""
            )
        }

    except Exception as e:

        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# =========================================================
# 12. START SERVER
# =========================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
