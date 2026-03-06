import logging
from typing import Optional

from .llm_client import create_llm_client

logger = logging.getLogger(__name__)


class AIAnalyzer:
    def __init__(self, config=None, api_key: Optional[str] = None):
        """
        Initialize the AI analyzer.

        Args:
            config: Config object (preferred method for initialization)
            api_key: Groq API key (deprecated, use config instead)
        """
        if config is None:
            # Legacy support: create minimal config from api_key
            from .config import Config
            config = Config()
            if api_key:
                config.config["groq_api_key"] = api_key

        self.client = create_llm_client(config)

    def get_analysis(self, report_content):
        """Get AI-powered analysis of the report"""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Analyze the following stock portfolio report and provide a brief summary of key insights, risks, and potential opportunities:\n\n{report_content}",
                    }
                ],
                model=self.client.model,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting AI analysis: {e}")
            return "AI analysis not available."
