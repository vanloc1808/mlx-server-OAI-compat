import logging
import time
from collections import defaultdict
from typing import Any, Dict

logger = logging.getLogger(__name__)

class RequestMetrics:
    """Handles metrics tracking for ML model requests."""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_time": 0,
            "request_types": defaultdict(int),
            "error_count": 0,
            "avg_tps": 0,
            "max_tps": 0,
            "min_tps": 0,
            "request_history": []
        }

    def update(self, request_type: str, metrics: Dict[str, Any]):
        """Update metrics with new request data."""
        self.metrics["total_requests"] += 1
        self.metrics["request_types"][request_type] += 1
        
        token_count = metrics.get("token_count", metrics.get("estimated_tokens", 0))
        elapsed_time = metrics.get("elapsed_time", 0)
        current_tps = metrics.get("tps", 0)
        
        self.metrics["total_tokens"] += token_count
        self.metrics["total_time"] += elapsed_time
        
        self._update_tps_metrics(current_tps)
        self._update_request_history(request_type, metrics)
        self._log_request_metrics(request_type, metrics)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        return {
            "total_requests": self.metrics["total_requests"],
            "total_tokens": self.metrics["total_tokens"],
            "total_time": self.metrics["total_time"],
            "request_types": dict(self.metrics["request_types"]),
            "error_count": self.metrics["error_count"],
            "performance": {
                "avg_tps": self.metrics["avg_tps"],
                "max_tps": self.metrics["max_tps"],
                "min_tps": self.metrics["min_tps"],
                "recent_requests": len(self.metrics["request_history"])
            }
        }

    def increment_error_count(self):
        """Increment the error counter."""
        self.metrics["error_count"] += 1

    @staticmethod
    def estimate_tokens(text: str) -> Dict[str, int]:
        """Estimate tokens in text with multiple methods."""
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        
        tokens_by_words = int(word_count / 1.3)
        tokens_by_chars = int(char_count / 4)
        estimated_tokens = (tokens_by_words + tokens_by_chars) // 2
        
        return {
            "estimated_tokens": estimated_tokens,
            "word_count": word_count,
            "char_count": char_count,
            "tokens_by_words": tokens_by_words,
            "tokens_by_chars": tokens_by_chars
        }

    def _update_tps_metrics(self, current_tps: float):
        total_requests = self.metrics["total_requests"]
        self.metrics["avg_tps"] = (self.metrics["avg_tps"] * (total_requests - 1) + current_tps) / total_requests
        self.metrics["max_tps"] = max(self.metrics["max_tps"], current_tps)
        if total_requests == 1 or current_tps < self.metrics["min_tps"]:
            self.metrics["min_tps"] = current_tps

    def _update_request_history(self, request_type: str, metrics: Dict[str, Any]):
        self.metrics["request_history"].append({
            "timestamp": time.time(),
            "request_type": request_type,
            **metrics
        })
        if len(self.metrics["request_history"]) > 100:
            self.metrics["request_history"].pop(0)

    def _log_request_metrics(self, request_type: str, metrics: Dict[str, Any]):
        logger.info(
            f"Request completed: {request_type}\n"
            f"Tokens: {metrics.get('token_count', 0)} (words: {metrics.get('word_count', 0)}, "
            f"chars: {metrics.get('char_count', 0)})\n"
            f"Time: {metrics.get('elapsed_time', 0):.2f}s\n"
            f"TPS: {metrics.get('tps', 0):.2f}\n"
            f"Avg TPS: {self.metrics['avg_tps']:.2f}"
        )
