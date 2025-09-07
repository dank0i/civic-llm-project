# agent.py
# Contains the actual agent code, responsible for actually running the various
# mitigations and reasonings, and formulating a final response.

from reasoning_framework import ReasoningFramework, ConversationContext, ReasoningStep, ConversationState
from bias_mitigation import BiasMitigationFramework, BiasDetector
from hallucination_prevention import HallucinationPreventer, UncertaintyCalibrator
from boundary_management import ConversationalIntelligence, ConversationFlowManager
from typing import List, Optional, Tuple

class PoliticalChatbotAgent:
    def __init__(self, api_key: Optional[str] = None, tavily_key: Optional[str] = None):
        self.context = ConversationContext()
        self.reasoning = ReasoningFramework()
        self.hallucination_preventer = HallucinationPreventer(api_key, tavily_key)
        
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
    
    async def validate_and_ground_response(
        self, 
        query: str, 
        response: str
    ) -> str:
        # Validates response and add uncertainty markers
        # Search for additional context if available
        search_results = await self.hallucination_preventer.validator.search_information(query)
        
        # Analyze information gaps
        gaps = self.hallucination_preventer.reason_about_information_gaps(
            query, 
            search_results
        )
        
        # Apply hallucination prevention
        validated = self.hallucination_preventer.prevent_hallucination(
            query,
            response
        )
        
        # Add confidence calibration
        confidence = self.hallucination_preventer.calibrator.calculate_confidence(
            "secondary",  # source quality
            7,  # days old
            "divided"  # consensus level
        )
        
        final_response = self.hallucination_preventer.calibrator.express_uncertainty(
            confidence,
            validated
        )
        
        # Record in reasoning chain
        self.create_reasoning_step(
            "hallucination_prevention",
            f"Applied uncertainty calibration (confidence: {confidence:.2f})",
            confidence
        )
        
        return final_response

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
        
        # Test biased text
        biased_text = "The radical left's policies are destroying the economy."
        
        print("Testing bias mitigation")
        print(f"Original: {biased_text}")
        
        mitigated, biases = agent.apply_bias_mitigation(biased_text)
        print(f"Mitigated: {mitigated}")
        print(f"Detected biases: {biases}")
    else:
        print("Set OPENAI_API_KEY to test bias mitigation")
        print("Bias mitigation framework structure created")

    print("Testing hallucination prevention framework")
    calibrator = UncertaintyCalibrator()
    
    test_confidence = calibrator.calculate_confidence(
        source_quality="secondary",
        temporal_relevance=15,
        consensus_level="divided"
    )
    
    print(f"Calculated confidence: {test_confidence:.2f}")
    
    test_statement = "The policy will have significant economic impact"
    uncertain_statement = calibrator.express_uncertainty(test_confidence, test_statement)
    
    print(f"Original: {test_statement}")
    print(f"With uncertainty: {uncertain_statement}")
    
    print("Hallucination prevention framework initialized")

    print("Testing Conversational Intelligence Framework")

    # Test boundary classification
    conv_intel = ConversationalIntelligence(api_key)
    
    test_queries = [
        "What's the weather today?",
        "Tell me about the debt ceiling",
        "How do I cook pasta?",
        "What are the economic implications of recent Fed decisions?"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        classification = conv_intel.classify_request(query, [])
        print(f"  In Scope: {classification['in_scope']}")
        print(f"  Confidence: {classification['confidence']}")
        
        if not classification['in_scope']:
            redirect = conv_intel.generate_graceful_redirect(query, classification)
            if redirect:
                print(f"  Redirect: {redirect[:100]}...")
    
    # Test conversation flow manager
    flow_manager = ConversationFlowManager()
    flow_manager.update_flow_state("test query", "normal", True)
    
    print("Conversational Intelligence Framework initialized")
    print(f"Engagement Score: {flow_manager.engagement_score}")