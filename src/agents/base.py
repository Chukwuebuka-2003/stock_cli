import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    """
    def __init__(self, name: str, model_client: Any = None):
        """
        Initialize the agent.
        
        Args:
            name: Name of the agent.
            model_client: Client for the LLM (e.g., Groq client).
        """
        self.name = name
        self.model_client = model_client
        
    @abstractmethod
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input context and return the result.
        
        Args:
            context: Dictionary containing input data and conversation history.
            
        Returns:
            Dictionary containing the agent's output.
        """
        pass

    def _log_step(self, message: str):
        """Helper to log agent steps."""
        logger.info(f"[{self.name}] {message}")
