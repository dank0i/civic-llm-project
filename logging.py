# logging.py
# 

import logging
import sys
from typing import Any, Dict, List, Optional
from datetime import datetime
from reasoning_framework import ReasoningStep
import json


class ChatbotLogger:
    # Structured logging system for the chatbot
    
    def __init__(self, name: str = "chatbot", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Create formatters
        self.detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        self.simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.simple_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for debugging
        self.file_handler = None
        
    def enable_file_logging(self, filename: str = "chatbot_debug.log"):
        # File logging for debugging
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)
        
        self.file_handler = logging.FileHandler(filename)
        self.file_handler.setFormatter(self.detailed_formatter)
        self.file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(self.file_handler)
        
        self.logger.info(f"File logging enabled: {filename}")
    
    def log_reasoning_step(self, step: ReasoningStep):
        # Log a reasoning step with structured data
        self.logger.info(f"Reasoning Step: {step.step_type}")
        self.logger.debug(f"  Content: {step.content[:100]}...")
        self.logger.debug(f"  Confidence: {step.confidence:.2f}")
        if step.perspectives:
            self.logger.debug(f"  Perspectives: {', '.join(step.perspectives)}")
        if step.uncertainties:
            self.logger.debug(f"  Uncertainties: {', '.join(step.uncertainties)}")
    
    def log_bias_detection(self, biases: List[str], corrected: bool = False):
        # Log bias detection events
        if biases:
            self.logger.warning(f"Biases detected: {', '.join(biases)}")
            if corrected:
                self.logger.info("Biases corrected successfully")
        else:
            self.logger.debug("No biases detected")
    
    def log_boundary_decision(self, query: str, in_scope: bool, confidence: float):
        # Log boundary management decisions
        scope_str = "IN SCOPE" if in_scope else "OUT OF SCOPE"
        self.logger.info(f"Boundary Decision: {scope_str} (confidence: {confidence:.2f})")
        self.logger.debug(f"  Query: {query[:50]}...")
    
    def log_conversation_state(self, old_state: str, new_state: str):
        # Log conversation state transitions
        self.logger.info(f"State Transition: {old_state} → {new_state}")
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        # Log performance metrics
        self.logger.info("Performance Metrics:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_error(self, error: Exception, context: str = ""):
        # Log errors with context
        self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)
    
    def set_verbosity(self, verbose: bool):
        # Toggle between verbose and normal logging
        level = logging.DEBUG if verbose else logging.INFO
        self.logger.setLevel(level)
        
        # Update console handler
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(level)
        
        self.logger.info(f"Logging verbosity set to: {'VERBOSE' if verbose else 'NORMAL'}")


class ConversationLogger:
    # Logger for conversation tracking
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_log = []
        self.logger = ChatbotLogger(f"Conversation_{self.session_id}")
    
    def log_exchange(self, user_message: str, bot_response: str, metadata: Dict[str, Any]):
        # Log a complete conversation exchange
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "bot": bot_response[:200] + "..." if len(bot_response) > 200 else bot_response,
            "metadata": metadata
        }
        
        self.conversation_log.append(exchange)
        
        self.logger.logger.info(f"Exchange #{len(self.conversation_log)}")
        self.logger.logger.debug(f"  User: {user_message[:50]}...")
        self.logger.logger.debug(f"  Bot: {bot_response[:50]}...")
        self.logger.logger.debug(f"  Confidence: {metadata.get('confidence', 'N/A')}")
    
    def save_conversation(self, filename: Optional[str] = None):
        # Save conversation log to file
        if not filename:
            filename = f"conversation_{self.session_id}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.conversation_log, f, indent=2)
        
        self.logger.logger.info(f"Conversation saved to {filename}")
    
    def get_summary(self) -> Dict[str, Any]:
        # Get conversation summary statistics
        return {
            "session_id": self.session_id,
            "total_exchanges": len(self.conversation_log),
            "start_time": self.conversation_log[0]["timestamp"] if self.conversation_log else None,
            "end_time": self.conversation_log[-1]["timestamp"] if self.conversation_log else None,
            "average_confidence": sum(
                e["metadata"].get("confidence", 0) for e in self.conversation_log
            ) / len(self.conversation_log) if self.conversation_log else 0
        }


class DebugTracer:
    # Debugging tracer for reasoning chains
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.trace_stack = []
        self.logger = ChatbotLogger("DebugTracer", logging.DEBUG if enabled else logging.WARNING)
    
    def enter_function(self, func_name: str, args: Dict[str, Any]):
        # Trace function entry
        if not self.enabled:
            return
        
        self.trace_stack.append(func_name)
        self.logger.logger.debug(f"→ Entering {func_name}")
        if args:
            self.logger.logger.debug(f"  Args: {json.dumps(args, default=str)[:100]}")
    
    def exit_function(self, func_name: str, result: Any = None):
        # Trace function exit
        if not self.enabled:
            return
        
        if self.trace_stack and self.trace_stack[-1] == func_name:
            self.trace_stack.pop()
        
        self.logger.logger.debug(f"← Exiting {func_name}")
        if result is not None:
            self.logger.logger.debug(f"  Result: {str(result)[:100]}")
    
    def log_decision_point(self, decision: str, choice: str, reason: str):
        # Log important decision points in reasoning
        if not self.enabled:
            return
        
        self.logger.logger.info(f"Decision: {decision}")
        self.logger.logger.info(f"  Choice: {choice}")
        self.logger.logger.info(f"  Reason: {reason}")
    
    def get_trace(self) -> List[str]:
        # Get current trace stack
        return self.trace_stack.copy()


def setup_logging(
    verbose: bool = False,
    debug: bool = False,
    log_file: Optional[str] = None
) -> ChatbotLogger:
    """
    Set up logging for the entire application
    
    Args:
        verbose: Enable verbose console output
        debug: Enable debug mode with detailed tracing
        log_file: Optional file for logging output
    
    Returns:
        Configured ChatbotLogger instance
    """
    # Determine logging level
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    # Create main logger
    main_logger = ChatbotLogger("PoliticalChatbot", level)
    
    # Enable file logging if specified
    if log_file:
        main_logger.enable_file_logging(log_file)
    
    # Configure root logger to capture library logs
    logging.getLogger().setLevel(level)
    
    # Suppress noisy library loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    main_logger.logger.info("Logging initialized")
    main_logger.logger.info(f"Log level: {logging.getLevelName(level)}")
    main_logger.logger.info(f"Debug mode: {debug}")
    main_logger.logger.info(f"Log file: {log_file or 'None'}")

    return main_logger


