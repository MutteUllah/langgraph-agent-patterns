"""Supervisor -> researcher -> analyst -> simulated action educational workflow."""

from __future__ import annotations

import argparse
import sys
from typing import Literal, TypedDict

from langchain.agents import create_agent
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from example_config import load_config
from tool_stubs import email_report, evaluate, web_search


class AgentState(TypedDict, total=False):
    goal: str
    current_step: int
    final_result: str
    next_agent: str
    research: str
    report: str


def supervisor_agent(
    state: AgentState,
) -> Command[Literal["research_agent", "analyst_agent", "action_agent", "__end__"]]:
    """Route once through the three stages, then finish."""
    if not state.get("research"):
        return Command(
            goto="research_agent",
            update={"next_agent": "research_agent", "current_step": 1},
        )
    if not state.get("report"):
        return Command(
            goto="analyst_agent",
            update={"next_agent": "analyst_agent", "current_step": 2},
        )
    if not state.get("final_result"):
        return Command(
            goto="action_agent",
            update={"next_agent": "action_agent", "current_step": 3},
        )
    return Command(goto=END, update={"next_agent": "finished"})


def _agent_text(result: dict) -> str:
    text = str(result["messages"][-1].text).strip()
    if not text:
        raise ValueError("An agent returned no text; the example cannot advance.")
    return text


def build_workflow(
    model, *, recipient: str = "reader@example.test", agent_factory=create_agent
):
    """Compile the intended graph. Model invocations happen only when it is run."""
    researcher = agent_factory(
        model=model,
        tools=[web_search],
        system_prompt=(
            "You are the research stage in an educational agent workflow. Draft brief research "
            "notes about the user's goal. The search tool is simulated and supplies no real "
            "sources; explicitly state this limitation. Do not claim to have browsed the web."
        ),
    )
    analyst = agent_factory(
        model=model,
        tools=[evaluate],
        system_prompt=(
            "Review the supplied research notes and produce a short, clear report. "
            "The review tool is a fixed demonstration stub. Preserve the simulated-search "
            "limitation, and do not present unsupported statements as verified research."
        ),
    )
    actor = agent_factory(
        model=model,
        tools=[email_report],
        system_prompt=(
            "Demonstrate the final email step using the email_report tool. It is simulated: "
            "no message can be sent. State that no email was sent and return a concise summary."
        ),
    )

    def research_agent(state: AgentState) -> Command[Literal["supervisor_agent"]]:
        result = researcher.invoke(
            {"messages": [("user", state["goal"])]}, config={"recursion_limit": 8}
        )
        return Command(
            update={"research": _agent_text(result)}, goto="supervisor_agent"
        )

    def analyst_agent(state: AgentState) -> Command[Literal["supervisor_agent"]]:
        result = analyst.invoke(
            {"messages": [("user", state["research"])]}, config={"recursion_limit": 8}
        )
        return Command(update={"report": _agent_text(result)}, goto="supervisor_agent")

    def action_agent(state: AgentState) -> Command[Literal["supervisor_agent"]]:
        result = actor.invoke(
            {
                "messages": [
                    ("user", f"Recipient: {recipient}\nReport:\n{state['report']}")
                ]
            },
            config={"recursion_limit": 8},
        )
        return Command(
            update={
                "final_result": "[Simulation: no email sent]\n" + _agent_text(result)
            },
            goto="supervisor_agent",
        )

    graph = StateGraph(AgentState)
    graph.add_node("supervisor_agent", supervisor_agent)
    graph.add_node("research_agent", research_agent)
    graph.add_node("analyst_agent", analyst_agent)
    graph.add_node("action_agent", action_agent)
    graph.add_edge(START, "supervisor_agent")
    return graph.compile()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("goal", help="Topic for the educational research workflow")
    parser.add_argument(
        "--recipient", default="reader@example.test", help="Simulated recipient only"
    )
    args = parser.parse_args(argv)
    if not args.goal.strip():
        parser.error("goal must not be blank")
    try:
        config = load_config()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    from langchain_google_genai import ChatGoogleGenerativeAI

    try:
        model = ChatGoogleGenerativeAI(
            model=config.chat_model,
            api_key=config.api_key,
            vertexai=False,
            temperature=0,
            max_retries=1,
            timeout=30,
        )
        workflow = build_workflow(model, recipient=args.recipient)
        result = workflow.invoke(
            {"goal": args.goal, "current_step": 0}, config={"recursion_limit": 12}
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary omits provider bodies and secrets.
        print(
            f"Example failed ({type(exc).__name__}). Check provider access, model configuration "
            "and the execution limits. No email integration is present.",
            file=sys.stderr,
        )
        return 1
    print(result["final_result"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
