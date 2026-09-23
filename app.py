import os
from typing import TypedDict

import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# --- 1. Graph State ---
class VerilogState(TypedDict):
    user_question: str
    generated_code: str
    critic_feedback: str
    critic_status: str
    attempt_count: int
    final_answer: str

MAX_ATTEMPTS = 3

def get_response_text(message) -> str:
    """Extract the plain answer text from an LLM response.

    Some Gemini models (when reasoning/thinking is enabled) return
    `message.content` as a LIST of blocks instead of a plain string, e.g.
    [{"type": "thinking", "thinking": "..."}, "the real answer text"].
    This pulls out only the real answer text and discards any thinking
    blocks, so downstream state/prompts/output never carry raw reasoning.
    """
    content = message.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "thinking":
                    continue
                text = block.get("text") or block.get("content")
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()

    return str(content).strip()


def strip_code_fence(text: str) -> str:
    """Strip a surrounding ```verilog / ``` markdown code fence, if present,
    so the final answer is plain Verilog source rather than markdown."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]  # drop opening ``` or ```verilog
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]  # drop closing ```
        return "\n".join(lines).strip()
    return stripped

# --- 2. Initialize Model ---
# Retrieve the key from the OS environment instead of Colab's userdata
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GOOGLE_API_KEY,
    temperature=0
)

# --- 3. Developer / Verilog Generator Agent ---
DEVELOPER_SYSTEM_PROMPT = """You are a senior digital design engineer who writes Verilog HDL.

Given a natural-language hardware requirement, you must:
- Understand the required functionality (combinational or sequential).
- Determine the correct module name, ports (inputs/outputs), internal
  registers/wires, clock, and reset behavior as appropriate.
- Write syntactically valid, clean, synthesizable Verilog.
- Use correct blocking (=) assignments in combinational always blocks and
  non-blocking (<=) assignments in sequential (clocked) always blocks.
- Handle edge cases (e.g. reset behavior, overflow/wraparound, default cases
  in case statements) sensibly.
- Parameterize the module when the requirement implies a variable width or
  configurable behavior.

When you receive critic feedback along with previously generated code, you
must fix every issue the critic raised while preserving anything that was
already correct. Do not ignore any point raised in the feedback.

Respond with ONLY the Verilog code (the module definition), with no prose
explanation before or after it. You may include brief `//` comments inside
the code itself.
"""

def developer_node(state: VerilogState) -> dict:
    question = state["user_question"]
    feedback = state.get("critic_feedback", "")
    previous_code = state.get("generated_code", "")

    if previous_code and feedback:
        user_msg = (
            f"Original requirement:\n{question}\n\n"
            f"Previously generated Verilog code:\n{previous_code}\n\n"
            f"Critic feedback (you MUST fix all of this):\n{feedback}\n\n"
            "Produce a corrected version of the Verilog code that fully "
            "addresses the feedback above."
        )
    else:
        user_msg = (
            f"Design requirement:\n{question}\n\n"
            "Generate the Verilog HDL code that implements this requirement."
        )

    messages = [
        ("system", DEVELOPER_SYSTEM_PROMPT),
        ("user", user_msg),
    ]
    response = llm_flash.invoke(messages)

    return {
        "generated_code": get_response_text(response),
        "attempt_count": state.get("attempt_count", 0) + 1,
    }

# --- 4. Critic Agent ---
CRITIC_SYSTEM_PROMPT = """You are a strict, senior Verilog design reviewer.

You review a developer's Verilog code against the original requirement with
zero tolerance for sloppy work. You are professional but blunt: you call out
every problem clearly, the way a demanding senior engineer would in a design
review. You never simply say "looks good" without real scrutiny.

Check specifically for:
- Incorrect Verilog syntax.
- Wrong module structure or missing/incorrect port declarations.
- Incorrect input/output/wire/reg declarations.
- Incorrect clock and reset handling (synchronous vs asynchronous, polarity).
- Wrong combinational vs sequential logic, or incorrect always block
  sensitivity lists.
- Incorrect use of blocking (=) vs non-blocking (<=) assignments.
- Incomplete implementation or missing functionality vs the requirement.
- Any mismatch between what the user asked for and what the hardware does.
- Potential synthesis issues (e.g. latches inferred unintentionally,
  incomplete case statements without a default).
- Missing edge cases (reset values, overflow/wraparound, boundary conditions).
- Missing or incorrect parameterization when the requirement implies it.

Respond in EXACTLY this format, with nothing else before or after it:

STATUS: ACCEPT
FEEDBACK: <one or two sentences confirming why it is correct>

or

STATUS: REJECT
FEEDBACK: <a precise, itemized explanation of every problem and exactly what
the developer must change to fix it>
"""

def critic_node(state: VerilogState) -> dict:
    question = state["user_question"]
    code = state["generated_code"]

    user_msg = (
        f"Original requirement:\n{question}\n\n"
        f"Generated Verilog code:\n{code}\n\n"
        "Review this code against the requirement now."
    )
    messages = [
        ("system", CRITIC_SYSTEM_PROMPT),
        ("user", user_msg),
    ]
    response = llm_flash.invoke(messages)
    text = get_response_text(response)

    status = "REJECT"
    feedback = text
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("STATUS:"):
            status = "ACCEPT" if "ACCEPT" in stripped.upper() else "REJECT"
        elif stripped.upper().startswith("FEEDBACK:"):
            feedback = stripped.split(":", 1)[1].strip()

    return {
        "critic_status": status,
        "critic_feedback": feedback,
    }

# --- 5. Routing / Decision ---
def route_after_critic(state: VerilogState) -> str:
    if state["critic_status"] == "ACCEPT":
        return "finalize"
    if state["attempt_count"] >= MAX_ATTEMPTS:
        return "finalize"
    return "developer"

# --- 6. Finalize Node ---
def finalize_node(state: VerilogState) -> dict:
    return {"final_answer": strip_code_fence(state["generated_code"])}

# --- 7. Build the LangGraph Workflow ---
graph_builder = StateGraph(VerilogState)
graph_builder.add_node("developer", developer_node)
graph_builder.add_node("critic", critic_node)
graph_builder.add_node("finalize", finalize_node)

graph_builder.set_entry_point("developer")
graph_builder.add_edge("developer", "critic")
graph_builder.add_conditional_edges(
    "critic",
    route_after_critic,
    {"developer": "developer", "finalize": "finalize"},
)
graph_builder.add_edge("finalize", END)

verilog_graph = graph_builder.compile()

# --- 8. Input/Output Formatting (LangServe I/O contract) ---
class AgentInput(BaseModel):
    input: str = Field(description="Your natural-language Verilog design question")


def format_for_graph(x) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input
    return {
        "user_question": user_input,
        "generated_code": "",
        "critic_feedback": "",
        "critic_status": "",
        "attempt_count": 0,
        "final_answer": "",
    }

def extract_final_answer(graph_output: dict) -> str:
    if isinstance(graph_output, dict):
        return graph_output.get("final_answer") or graph_output.get("generated_code", str(graph_output))
    return str(graph_output)

formatted_agent_chain = (
    RunnableLambda(format_for_graph)
    | verilog_graph
    | RunnableLambda(extract_final_answer)
).with_types(input_type=AgentInput, output_type=str)

# --- 9. FastAPI App ---
app = FastAPI()
add_routes(app, formatted_agent_chain, path="/agent")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
