from typing import Dict, Any
from .base import BaseAgent

class AnalysisAgent(BaseAgent):
    """
    Agent responsible for analyzing research findings.
    """
    def __init__(self, model_client):
        super().__init__("AnalysisAgent", model_client)

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the findings.
        """
        query = context.get("query")
        findings = context.get("findings", [])
        
        self._log_step("Analyzing findings...")
        
        findings_text = "\n".join([f"Step {f['step_id']} ({f['description']}): {f['result']}" for f in findings])
        
        system_prompt = """
        You are an Analysis Agent. Your job is to analyze the raw data gathered by the Research Agent.
        Identify key trends, correlations, and important facts relevant to the user's query.
        Be objective and data-driven.
        """
        
        user_prompt = f"""
        User Query: {query}
        
        Research Findings:
        {findings_text}
        
        Provide a detailed analysis of these findings.
        """
        
        try:
            response = self.model_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            analysis = response.choices[0].message.content
            return {"analysis": analysis}
            
        except Exception as e:
            self._log_step(f"Error analyzing findings: {e}")
            return {"analysis": "Failed to generate analysis."}
