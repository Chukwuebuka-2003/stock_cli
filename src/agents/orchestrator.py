import logging
from typing import Dict, Any
from .planner import PlanningAgent
from .researcher import ResearchAgent
from .analyst import AnalysisAgent
from .decision import DecisionAgent

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """
    Orchestrates the multi-agent workflow.
    """
    def __init__(
        self,
        model_client,
        vector_store=None,
        tavily_api_key=None,
        data_fetcher=None,
    ):
        self.planner = PlanningAgent(model_client)
        self.researcher = ResearchAgent(
            model_client,
            vector_store,
            tavily_api_key=tavily_api_key,
            data_fetcher=data_fetcher,
        )
        self.analyst = AnalysisAgent(model_client)
        self.decision_maker = DecisionAgent(model_client)
        
    def run(self, query: str) -> str:
        """
        Run the full agent workflow for a given query.
        """
        logger.info(f"Starting agent workflow for query: {query}")
        
        context = {"query": query}
        
        # 1. Plan
        plan_result = self.planner.process(context)
        context.update(plan_result)
        self._ensure_web_context(query, context)
        
        # 2. Research
        research_result = self.researcher.process(context)
        context.update(research_result)
        
        # 3. Analyze
        analysis_result = self.analyst.process(context)
        context.update(analysis_result)
        
        # 4. Decide
        decision_result = self.decision_maker.process(context)
        
        return decision_result.get("response", "No response generated.")

    @staticmethod
    def _ensure_web_context(query: str, context: Dict[str, Any]) -> None:
        """Guarantee at least one Tavily-enabled search for broader context."""
        plan = context.get("plan", [])
        has_web = any(step.get("tool") == "search_web" for step in plan)
        if has_web:
            return

        next_step_id = max([step.get("step_id", 0) for step in plan] or [0]) + 1
        plan.append(
            {
                "step_id": next_step_id,
                "description": "Run a quick web scan for the latest news and sentiment.",
                "tool": "search_web",
                "query": query,
            }
        )
        context["plan"] = plan
