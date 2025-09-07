# agent.py
# 

from reasoning_framework import ReasoningFramework, ConversationContext, ReasoningStep, ConversationState
from bias_mitigation import BiasMitigationFramework, BiasDetector
from typing import List, Dict, Optional, Tuple

class PoliticalChatbotAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.context = ConversationContext()
        self.reasoning = ReasoningFramework()
        
        if api_key:
            self.bias_detector = BiasDetector(api_key)
        else:
            self.bias_detector = None
    
    def create_reasoning_step(
        self, 
        step_type: str, 
        content: str, 
        confidence: float = 0.5
    ) -> ReasoningStep:
        step = ReasoningStep(
            step_type=step_type,
            content=content,
            confidence=confidence
        )
        self.context.reasoning_chain.append(step)
        return step
    
    def update_conversation_state(self, new_state: ConversationState):
        old_state = self.context.state
        self.context.state = new_state
    
    def apply_bias_mitigation(self, response: str) -> Tuple[str, List[str]]:
        # Apply bias detection and mitigation to a response

        if not self.bias_detector:
            return response, []
        
        # Apply bias mitigation
        mitigated_response, detected_biases = self.bias_detector.detect_mitigate_bias(response)
        
        # Record detected biases
        self.context.detected_biases.extend(detected_biases)
        
        # Create reasoning step
        self.create_reasoning_step(
            "bias_mitigation",
            f"Detected and corrected {len(detected_biases)} potential biases",
            0.8
        )
        
        return mitigated_response, detected_biases

if __name__ == "__main__":
    import os
    
    # Test bias mitigation framework
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        agent = PoliticalChatbotAgent(api_key)

        # Test reasoning step creation
        step = agent.create_reasoning_step(
            "information_validation",
            "Testing basic reasoning chain creation",
            0.75
        )

        # Test to ensure program and structures initiate properly
        print("Core Reasoning Architecture")
        print(f"Created reasoning step: {step.step_type}")
        print("Reasoning frameworks available:")
        print("- Information Validation")
        print("- Perspective Taking")
        print("- Uncertainty Calibration")
        
        # Test biased text
        biased_text = "The radical left's policies are destroying the economy."
        
        print("Testing bias mitigation...")
        print(f"Original: {biased_text}")
        
        mitigated, biases = agent.apply_bias_mitigation(biased_text)
        print(f"Mitigated: {mitigated}")
        print(f"Detected biases: {biases}")
    else:
        print("Set OPENAI_API_KEY to test bias mitigation")
        print("Bias mitigation framework structure created")