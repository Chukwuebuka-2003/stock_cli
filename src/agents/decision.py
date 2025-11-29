import json
from typing import Any, Dict, List

from .base import BaseAgent

class DecisionAgent(BaseAgent):
    """
    Agent responsible for making final recommendations and formatting the response.
    """
    def __init__(self, model_client):
        super().__init__("DecisionAgent", model_client)

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formulate the final answer.
        """
        query = context.get("query", "")
        analysis = context.get("analysis", "")
        findings: List[Dict[str, Any]] = context.get("findings", [])
        plan_steps: List[Dict[str, Any]] = context.get("plan", [])
        quotes: Dict[str, Any] = context.get("quotes", {})
        
        self._log_step("Formulating final decision/response...")
        
        system_prompt = (
            "You are the Decision Agent for an investing copilot. Summarize the findings and analysis into JSON with this schema: {"
            "\"market_snapshot\": string, \"web_highlights\": [string], \"ai_takeaways\": [string], \"action_items\": [string]}.")
        system_prompt += (
            "Each string should be concise (<= 2 sentences) and grounded only in the supplied findings/analysis. "
            "If there is no data for a section, return an empty string (for market_snapshot) or an empty list (for arrays). "
            "Always respond with valid JSON only."
        )

        findings_text = "\n".join(
            [
                f"Step {f['step_id']} | Tool: {f.get('tool')} | {f['description']} -> {f['result']}"
                for f in findings
            ]
        ) or "No findings were gathered."

        plan_text = "\n".join(
            [
                f"Step {p['step_id']}: {p['description']} ({p['tool']})"
                for p in plan_steps
            ]
        ) or "No plan provided."

        if quotes:
            price_lines = []
            for symbol, data in quotes.items():
                summary = data.get("summary")
                if summary:
                    price_lines.append(summary)
                else:
                    price = data.get("price")
                    line = f"{symbol}: ${price:,.2f}" if isinstance(price, (int, float)) else symbol
                    price_lines.append(line)
            price_block = "\n".join(price_lines)
        else:
            price_block = "No live quote data collected."

        user_prompt = f"""
        User Query: {query}

        Plan Overview:
        {plan_text}

        Price Data:
        {price_block}

        Research Findings:
        {findings_text}

        Analysis Summary:
        {analysis or 'No analysis available.'}

        Compose the final response following the mandated template.
        """
        
        try:
            response = self.model_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            raw_content = response.choices[0].message.content
            try:
                payload = json.loads(raw_content)
            except json.JSONDecodeError:
                return {"response": raw_content}

            if quotes:
                payload["market_snapshot"] = price_block
            formatted = self._format_response(payload)
            return {"response": formatted}
        
        except Exception as e:
            self._log_step(f"Error formulating decision: {e}")
            return {"response": "Failed to generate final response."}

    @staticmethod
    def _format_response(payload: Dict[str, Any]) -> str:
        """Convert the JSON payload into consistent Markdown."""
        market_snapshot = payload.get("market_snapshot") or ""
        web_highlights = payload.get("web_highlights") or []
        ai_takeaways = payload.get("ai_takeaways") or []
        action_items = payload.get("action_items") or []

        sections: List[str] = []

        if market_snapshot:
            lines = [line.strip() for line in str(market_snapshot).splitlines() if line.strip()]
            if lines:
                bullets = "\n".join(f"- {line}" for line in lines)
                sections.append("## Market Snapshot\n" + bullets)

        if web_highlights:
            bullets = "\n".join(f"- {item}" for item in web_highlights if item)
            if bullets:
                sections.append("## Web Highlights\n" + bullets)

        if ai_takeaways:
            bullets = "\n".join(f"- {item}" for item in ai_takeaways if item)
            if bullets:
                sections.append("## AI Takeaways\n" + bullets)

        if action_items:
            bullets = "\n".join(f"- {item}" for item in action_items if item)
            if bullets:
                sections.append("## Action Items\n" + bullets)

        if not sections:
            return "No actionable insights were generated."

        return "\n\n".join(sections)
