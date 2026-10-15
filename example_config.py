"""Explicit configuration; importing examples never loads secrets or builds clients."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class ExampleConfig:
    api_key: str = field(repr=False)
    chat_model: str
    embedding_model: str = ""
    embedding_dimensions: int = 3072


def config_from_environment(
    values: Mapping[str, str], *, require_embeddings: bool = False
) -> ExampleConfig:
    """Validate required values without constructing or calling a model client."""
    required = ["GEMINI_API_KEY", "GEMINI_CHAT_MODEL"]
    if require_embeddings:
        required.append("GEMINI_EMBEDDING_MODEL")
    missing = [key for key in required if not values.get(key, "").strip()]
    if missing:
        raise ValueError(
            "Missing configuration: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and set these values."
        )
    try:
        dimensions = int(values.get("GEMINI_EMBEDDING_DIMENSIONS", "3072"))
        if dimensions <= 0:
            raise ValueError
    except ValueError:
        raise ValueError(
            "GEMINI_EMBEDDING_DIMENSIONS must be a positive integer."
        ) from None
    return ExampleConfig(
        api_key=values["GEMINI_API_KEY"].strip(),
        chat_model=values["GEMINI_CHAT_MODEL"].strip(),
        embedding_model=values.get("GEMINI_EMBEDDING_MODEL", "").strip(),
        embedding_dimensions=dimensions,
    )


def load_config(*, require_embeddings: bool = False) -> ExampleConfig:
    """Load this project's optional .env only when a CLI is explicitly run."""
    load_dotenv(Path(__file__).with_name(".env"), override=False)
    return config_from_environment(os.environ, require_embeddings=require_embeddings)
