# reasoning_framework.py
# Contains the reasoning architecture for the chatbot, including data structures for 
# current states and steps in the conversation, validating, and reasoning.

from typing import List, Dict
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

class ConversationState(Enum):
    # Represents the current state of the conversation
    INITIAL = "initial"
    POLITICAL_DISCUSSION = "political_discussion"
    BOUNDARY_CROSSED = "boundary_crossed"
    INFORMATION_SYNTHESIS = "information_synthesis"
    PERSPECTIVE_ANALYSIS = "perspective_analysis"

@dataclass
class ReasoningStep:
    # Represents a single step in the reasoning chain
    step_type: str
    content: str
    confidence: float
    sources: List[str] = field(default_factory=list)
    perspectives: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)

@dataclass
class ConversationContext:
    # Maintains conversation state and history
    state: ConversationState = ConversationState.INITIAL
    message_history: deque = field(default_factory=lambda: deque(maxlen=10))
    reasoning_chain: List[ReasoningStep] = field(default_factory=list)
    detected_biases: List[str] = field(default_factory=list)
    current_perspectives: Dict[str, str] = field(default_factory=dict)
    confidence_level: float = 1.0
    scope_violations: List[str] = field(default_factory=list)


class ReasoningFramework:
    # Multi-step reasoning chains for political information validation
    @staticmethod
    def get_information_validation_chain() -> str:
        return """
        Reasoning chain for information validation:
        1. Assess claim responsibility:
           - Does this align with known political processes?
           - Are there logical inconsistencies?
           - What is the coherence score (0-1)?
        
        2. Identify information and knowledge gaps:
           - What key facts are missing?
           - What context is needed for understanding?
           - What assumptions am I making?
        
        3. Quantify uncertainity appropraitely 
           - What is my confidence level for each claim?
           - Which parts require external validation?
           - How do I express appropriate uncertainty?
        
        4. Highlight caveats and knowledge gaps
           - Present verified information clearly
           - Acknowledge unverified claims explicitly
           - Provide reasoning for confidence levels
        """
    
    @staticmethod
    def get_perspective_taking_framework() -> str:
        return """
        Perspective analysis framework:
        1. Identify major stakeholders:
           - Who are the primary actors?
           - What are their stated positions?
           - What are their underlying interests?
        
        2. Map viewpoint and perspective dimensions:
           - Political ideology spectrum
           - Economic implications
           - Social/cultural considerations
           - Legal/constitutional aspects
           - Practical/implementation concerns
        
        3. Analyze sources and reasoning perspectives
           - What evidence does each side cite?
           - What are the logical foundations?
           - Where do interpretations diverge?
        
        4. Synthesize balanced view
           - Present each perspective fairly
           - Avoid false equivalences
           - Acknowledge complexity and nuance
        """
    
    @staticmethod
    def get_uncertainty_calibration_protocol() -> str:
        return """
        Confidence calibration protocol:
        1. Analyze information source quality:
           - Primary sources: HIGH confidence (0.8-1.0)
           - Secondary analysis: MEDIUM confidence (0.5-0.8)
           - Speculation/inference: LOW confidence (0.2-0.5)
           - Unverified claims: MINIMAL confidence (0.0-0.2)
        
        2. Analyze temporal relevance:
           - Current (< 1 week): No degradation
           - Recent (1 week - 1 month): -0.1 confidence
           - Dated (1-6 months): -0.2 confidence
           - Historical (> 6 months): -0.3 confidence
        
        3. Analyze consensus agreement:
           - Universal agreement: +0.2 confidence
           - Broad consensus: +0.1 confidence
           - Contested/debated: -0.1 confidence
           - Highly controversial: -0.2 confidence
        
        4. Express uncertainty appropriately:
           - High confidence: "Based on verified sources..."
           - Medium confidence: "Available evidence suggests..."
           - Low confidence: "While unconfirmed, some reports indicate..."
           - Minimal confidence: "I cannot verify this, but..."
        """
