"""
Orchestrator: implements the "Complete Agent Pipeline" from the architecture doc.

    User Request
      -> Input Processing
      -> Reasoning Engine
      -> Planning Agent
      -> Reactive Agent (if simple)
      -> RAG Retrieval
      -> Tool Execution
      -> Multi-Agent Collaboration
      -> Autonomous Workflow
      -> Memory Update
      -> Final Response
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .agents.multi_agent import MultiAgentCoordinator, MultiAgentReport
from .agents.planning import PlanningAgent, PlanStep
from .agents.rag import RAGAgent
from .agents.reactive import ReactiveAgent, build_default_reactive_agent
from .agents.tool_using import ToolUsingAgent
from .agents.workflow import AutonomousWorkflowManager, WorkflowResult
from .input_processing import ConversationContext, InputProcessor
from .llm import LLMBackend, get_default_backend
from .memory import MemoryManager
from .reasoning_engine import ReasoningEngine


@dataclass
class PipelineTrace:
    """Records what happened at each stage, for transparency/debugging."""

    intent: Optional[str] = None
    handled_reactively: bool = False
    reactive_output: Optional[str] = None
    plan: List[PlanStep] = field(default_factory=list)
    retrieved_context: List[str] = field(default_factory=list)
    tool_used: Optional[str] = None
    tool_output: Optional[str] = None
    multi_agent_report: Optional[MultiAgentReport] = None
    workflow_result: Optional[WorkflowResult] = None
    final_response: Optional[str] = None


class Orchestrator:
    def __init__(
        self,
        llm: Optional[LLMBackend] = None,
        long_term_memory_path: str = "long_term_memory.json",
        reactive_agent: Optional[ReactiveAgent] = None,
        multi_agent_pipeline: Optional[List[str]] = None,
        workflow_max_retries: int = 2,
    ):
        self.llm = llm or get_default_backend()

        self.input_processor = InputProcessor(self.llm)
        self.reasoning_engine = ReasoningEngine(self.llm)
        self.reactive_agent = reactive_agent or build_default_reactive_agent()
        self.planning_agent = PlanningAgent(self.llm)
        self.tool_agent = ToolUsingAgent(self.llm)
        self.rag_agent = RAGAgent(self.llm)
        self.multi_agent = MultiAgentCoordinator(self.llm, pipeline=multi_agent_pipeline)
        self.workflow_manager = AutonomousWorkflowManager(max_retries=workflow_max_retries)

        self.memory = MemoryManager(long_term_path=long_term_memory_path)
        self.context = ConversationContext()

    def add_knowledge(self, doc_id: str, text: str) -> None:
        """Feed a document into the RAG agent's knowledge base."""
        self.rag_agent.add_document(doc_id, text)

    def handle(self, user_request: str) -> PipelineTrace:
        trace = PipelineTrace()

        # 1. Input Processing
        processed = self.input_processor.process(user_request, self.context)
        request = processed["request"]
        trace.intent = processed["intent"]
        self.memory.short_term.add_turn(f"user: {request}")

        # 2. Reasoning Engine decides: reactive vs planned
        if self.reasoning_engine.should_react(request):
            reactive_output = self.reactive_agent.handle(request)
            if reactive_output:
                trace.handled_reactively = True
                trace.reactive_output = reactive_output
                trace.final_response = reactive_output
                self.memory.short_term.add_turn(f"assistant: {reactive_output}")
                return trace

        # 3. Planning Agent produces a task list for non-trivial goals
        trace.plan = self.planning_agent.plan(request)
        self.memory.short_term.add_task(request)

        # 4. RAG retrieval against the knowledge base
        trace.retrieved_context = self.rag_agent.retrieve(request)

        # 5. Tool execution, if the reasoning engine decides a tool is needed
        tool_result = self.tool_agent.maybe_run_tool(request)
        trace.tool_used = tool_result["tool_used"]
        trace.tool_output = tool_result["output"]

        # 6. Multi-agent collaboration for complex build/create tasks
        if trace.intent == "Create Software Project":
            trace.multi_agent_report = self.multi_agent.run(request)

        # 7. Autonomous workflow wraps the "expensive" work with retry/evaluation
        def execute() -> str:
            working_context_parts = []
            if trace.plan:
                working_context_parts.append(
                    "Plan:\n" + "\n".join(f"{s.index}. {s.description}" for s in trace.plan)
                )
            if trace.retrieved_context:
                working_context_parts.append(
                    "Retrieved context:\n" + "\n".join(trace.retrieved_context)
                )
            if trace.tool_output:
                working_context_parts.append(f"Tool ({trace.tool_used}) output:\n{trace.tool_output}")
            if trace.multi_agent_report:
                working_context_parts.append("Sub-agent results:\n" + trace.multi_agent_report.summary())

            working_context = "\n\n".join(working_context_parts)
            return self.reasoning_engine.synthesize(request, working_context)

        trace.workflow_result = self.workflow_manager.run(request, execute_fn=execute)
        trace.final_response = trace.workflow_result.output or "I couldn't produce a satisfactory answer."

        # 8. Memory update
        self.memory.short_term.add_turn(f"assistant: {trace.final_response}")
        if trace.intent == "Create Software Project":
            self.memory.long_term.record_project(request)

        return trace
