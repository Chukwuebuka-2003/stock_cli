import logging

from groq import Groq

logger = logging.getLogger(__name__)


class AIAnalyzer:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

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
                model="llama-3.3-70b-versatile",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting AI analysis: {e}")
            return "AI analysis not available."
