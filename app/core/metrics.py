import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)

class RequestMetrics:
    """Handles metrics tracking for ML model requests."""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "tps": 0,  # Tokens per second
            "ttft": 0,  # Time to first token
            "throughput": 0,  # Request throughput 
            "error_count": 0
        }

    def update(self, request_type: str, metrics: Dict[str, Any]):
        """Update metrics with new request data."""
        self.metrics["total_requests"] += 1
        
        # Extract key performance metrics
        current_tps = metrics.get("tps", 0)
        current_ttft = metrics.get("ttft", 0)
        current_throughput = metrics.get("throughput", 0)
        
        # Update rolling averages
        total_requests = self.metrics["total_requests"]
        if total_requests > 1:
            self.metrics["tps"] = (self.metrics["tps"] * (total_requests - 1) + current_tps) / total_requests
            self.metrics["ttft"] = (self.metrics["ttft"] * (total_requests - 1) + current_ttft) / total_requests
            self.metrics["throughput"] = (self.metrics["throughput"] * (total_requests - 1) + current_throughput) / total_requests
        else:
            self.metrics["tps"] = current_tps
            self.metrics["ttft"] = current_ttft
            self.metrics["throughput"] = current_throughput
        
        self._log_request_metrics(request_type, metrics)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        return {
            "total_requests": self.metrics["total_requests"],
            "performance": {
                "tps": self.metrics["tps"],
                "ttft": self.metrics["ttft"],
                "throughput": self.metrics["throughput"]
            },
            "error_count": self.metrics["error_count"]
        }

    def increment_error_count(self):
        """Increment the error counter."""
        self.metrics["error_count"] += 1

    @staticmethod
    def estimate_tokens(text: str) -> Dict[str, int]:
        """Estimate tokens in text with a simplified method."""
        char_count = len(text)
        estimated_tokens = int(char_count / 4)  # Simple character-based estimation
        
        return {
            "estimated_tokens": estimated_tokens
        }

    def _log_request_metrics(self, request_type: str, metrics: Dict[str, Any]):
        """Log simplified key metrics."""
        logger.info(
            f"Request completed: {request_type}\n"
            f"TPS: {metrics.get('tps', 0):.2f}\n"
            f"TTFT: {metrics.get('ttft', 0):.2f}ms\n"
            f"Throughput: {metrics.get('throughput', 0):.2f} req/s"
        )
