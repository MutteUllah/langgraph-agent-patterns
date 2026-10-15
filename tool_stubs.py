"""Deterministic educational tools. None connects to an external service."""

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Return a simulated search response; this tool does not browse the internet."""
    return (
        "SIMULATED SEARCH: No internet search was performed. "
        "Use this placeholder to explain the research workflow; no factual sources are supplied."
    )


@tool
def evaluate(research: str) -> str:
    """Return a fixed demonstration review, not an independent factual evaluation."""
    return (
        "SIMULATED REVIEW: Check clarity, identify unsupported claims, "
        "and state that the research uses placeholder search data."
    )


@tool
def email_report(recipient: str, report: str) -> str:
    """Simulate an email action. No email is sent, queued, or saved."""
    return "SIMULATED EMAIL: No email was sent. This example has no email integration."


@tool
def check_account(account_id: int) -> str:
    """Return fictional account data; no financial service or real account is queried."""
    return (
        "SIMULATED ACCOUNT: balance=1000.00 USD; debt=0.00 USD. "
        "These are fictional demonstration values; no account system was queried."
    )
