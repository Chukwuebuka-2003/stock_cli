import json
from typing import Dict, Any, List
from .base import BaseAgent

class PlanningAgent(BaseAgent):
    """
    Agent responsible for analyzing the user's request and creating a research plan.
    """
    def __init__(self, model_client):
        super().__init__("PlanningAgent", model_client)

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the query and generate a plan.
        """
        query = context.get("query")
        self._log_step(f"Analyzing query: {query}")
        
        system_prompt = """
        You are a Planning Agent for a stock analysis tool. 
        Your goal is to break down a user's financial query into specific research steps.
        
        Available tools for the Research Agent:
        1. search_web(query): Search the internet for news and information.
        2. search_rag(query): Search local knowledge base (documents, reports).
        3. get_stock_price(symbol): Get current stock price.
        
        Output a JSON object with a 'plan' key containing a list of steps. 
        Each step should have:
        - 'step_id': integer
        - 'description': string
        - 'tool': string (one of the available tools)
        - 'query': string (the input for the tool)
        
        Example:
        User: "Analyze Apple's latest earnings and stock performance."
        Output:
        {
            "plan": [
                {"step_id": 1, "description": "Get current stock price for AAPL", "tool": "get_stock_price", "query": "AAPL"},
                {"step_id": 2, "description": "Search for latest earnings report analysis", "tool": "search_web", "query": "Apple latest earnings report analysis"},
                {"step_id": 3, "description": "Search local documents for historical context", "tool": "search_rag", "query": "Apple earnings history"}
            ]
        }
        """
        
        try:
            response = self.model_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                response_format={"type": "json_object"}
            )
            
            plan_json = response.choices[0].message.content
            plan = json.loads(plan_json)
            
            self._log_step(f"Generated plan with {len(plan.get('plan', []))} steps")
            return {"plan": plan.get("plan", [])}
            
        except Exception as e:
            self._log_step(f"Error generating plan: {e}")
            # Fallback plan
            return {"plan": [{"step_id": 1, "description": "Search web for query", "tool": "search_web", "query": query}]}
