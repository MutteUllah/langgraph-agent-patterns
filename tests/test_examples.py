"""Offline regression coverage: no API credentials or provider requests required."""

from __future__ import annotations

import contextlib
import io
import sqlite3
import unittest
from unittest.mock import patch

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END
from langgraph.store.sqlite import SqliteStore

import a
import project
from example_config import config_from_environment
from tool_stubs import check_account, email_report, evaluate, web_search


class ConstantEmbeddings(Embeddings):
    """Small offline vectors to exercise the actual SQLite vector path."""

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0, 0.0]


class OfflineStructuredModel:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, messages):
        if self.schema is project.Plan:
            return project.Plan(
                plan_steps=["Read the request", "Explain the simulated result"]
            )
        return project.ExtractorSchema(facts=["The user's name is Ada."])


class OfflineToolModel:
    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return OfflineStructuredModel(schema)

    def invoke(self, messages):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="These account values are simulated, Ada.")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "check_account",
                    "args": {"account_id": 1},
                    "id": "offline-call",
                }
            ],
        )


class ConfigTests(unittest.TestCase):
    def test_missing_key_and_model_fail_without_showing_values(self):
        with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY, GEMINI_CHAT_MODEL"):
            config_from_environment({})

    def test_embedding_config_is_required_only_for_memory_example(self):
        values = {
            "GEMINI_API_KEY": "offline-test-value",
            "GEMINI_CHAT_MODEL": "test-model",
        }
        config_from_environment(values)
        with self.assertRaisesRegex(ValueError, "GEMINI_EMBEDDING_MODEL"):
            config_from_environment(values, require_embeddings=True)
        self.assertNotIn("offline-test-value", repr(config_from_environment(values)))

    def test_invalid_dimensions_fail_before_client_creation(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            config_from_environment(
                {
                    "GEMINI_API_KEY": "test",
                    "GEMINI_CHAT_MODEL": "test",
                    "GEMINI_EMBEDDING_DIMENSIONS": "0",
                }
            )

    def test_cli_reports_missing_config_before_running_any_workflow(self):
        for module in (a, project):
            with self.subTest(example=module.__name__):
                stderr = io.StringIO()
                with (
                    patch.object(
                        module,
                        "load_config",
                        side_effect=ValueError("Missing configuration"),
                    ),
                    patch.object(module, "build_workflow") as build,
                    contextlib.redirect_stderr(stderr),
                ):
                    self.assertEqual(module.main(["test goal"]), 2)
                build.assert_not_called()
                self.assertIn("Configuration error:", stderr.getvalue())


class ToolTests(unittest.TestCase):
    def test_stub_results_are_explicitly_simulated(self):
        cases = [
            (web_search, {"query": "test"}),
            (evaluate, {"research": "test"}),
            (email_report, {"recipient": "reader@example.test", "report": "test"}),
            (check_account, {"account_id": 1}),
        ]
        for stub, args in cases:
            with self.subTest(tool=stub.name):
                self.assertIn("SIMULATED", stub.invoke(args))
        self.assertIn("No email was sent", email_report.invoke(cases[2][1]))
        self.assertEqual(
            check_account.invoke({"account_id": 1}),
            check_account.invoke({"account_id": 2}),
        )


class SupervisorTests(unittest.TestCase):
    def test_supervisor_selects_each_stage_and_then_ends(self):
        states = [
            ({"goal": "test"}, "research_agent"),
            ({"research": "notes"}, "analyst_agent"),
            ({"research": "notes", "report": "report"}, "action_agent"),
            ({"research": "notes", "report": "report", "final_result": "done"}, END),
        ]
        for state, expected in states:
            self.assertEqual(a.supervisor_agent(state).goto, expected)

    def test_compiled_supervisor_finishes_all_three_stages(self):
        calls = []

        class StubAgent:
            def __init__(self, label):
                self.label = label

            def invoke(self, payload, config):
                self_payload = payload["messages"][0]
                calls.append((self.label, self_payload))
                return {"messages": [AIMessage(content=f"{self.label} output")]}

        def factory(**kwargs):
            return StubAgent(kwargs["tools"][0].name)

        graph = a.build_workflow(object(), agent_factory=factory)
        result = graph.invoke({"goal": "test goal"}, config={"recursion_limit": 12})
        self.assertEqual(
            [call[0] for call in calls], ["web_search", "evaluate", "email_report"]
        )
        self.assertEqual(result["next_agent"], "finished")
        self.assertEqual(result["current_step"], 3)
        self.assertTrue(
            result["final_result"].startswith("[Simulation: no email sent]")
        )
        self.assertEqual(calls[1][1][1], "web_search output")

    def test_actual_create_agent_accepts_the_stage_message_payloads(self):
        class OfflineChatModel(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        model = OfflineChatModel(
            responses=[
                AIMessage(content="Simulated research notes"),
                AIMessage(content="Simulated report"),
                AIMessage(content="No email was sent"),
            ]
        )
        result = a.build_workflow(model).invoke({"goal": "test"})
        self.assertEqual(result["next_agent"], "finished")
        self.assertIn("No email was sent", result["final_result"])


class MemoryWorkflowTests(unittest.TestCase):
    def test_tool_loop_memory_namespace_and_real_sqlite_checkpoint_offline(self):
        with (
            contextlib.closing(
                sqlite3.connect(
                    ":memory:", check_same_thread=False, isolation_level=None
                )
            ) as memory,
            contextlib.closing(
                sqlite3.connect(":memory:", check_same_thread=False)
            ) as checkpoints,
        ):
            store = SqliteStore(
                memory,
                index={"embed": ConstantEmbeddings(), "dims": 4, "fields": ["fact"]},
            )
            store.setup()
            saver = SqliteSaver(checkpoints)
            saver.setup()
            graph = project.build_workflow(
                OfflineToolModel(), checkpointer=saver, store=store
            )
            config = {"configurable": {"thread_id": "offline-thread", "user_id": "ada"}}
            result = graph.invoke(
                {"messages": [("user", "My name is Ada. Show simulated account 1.")]},
                config=config,
            )
            self.assertEqual(result["steps"], 2)
            self.assertTrue(
                any(isinstance(msg, ToolMessage) for msg in result["messages"])
            )
            self.assertEqual(
                result["messages"][-1].content,
                "These account values are simulated, Ada.",
            )
            self.assertEqual(
                store.search(("ada", "semantic"), query="name")[0].value["fact"],
                "The user's name is Ada.",
            )
            self.assertEqual(store.search(("other-user", "semantic")), [])
            self.assertIsNotNone(saver.get_tuple(config))


if __name__ == "__main__":
    unittest.main()
