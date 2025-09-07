# agent.py
# Contains the actual agent code, responsible for actually running the various
# mitigations and reasonings, and formulating a final response.

from reasoning_framework import ReasoningFramework, ConversationContext, ReasoningStep, ConversationState
from bias_mitigation import BiasMitigationFramework, BiasDetector, PerspectiveAnalysisFramework
from hallucination_prevention import HallucinationPreventer, UncertaintyCalibrator
from boundary_management import ConversationalIntelligence, ConversationFlowManager
from typing import List, Optional, Tuple
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoliticalChatbotAgent:
    def __init__(self, api_key: Optional[str] = None, tavily_key: Optional[str] = None):
        self.context = ConversationContext()
        self.reasoning = ReasoningFramework()
        logger.info("Initialized agent with reasoning framework")
        
        self.hallucination_preventer = HallucinationPreventer(api_key, tavily_key)
        logger.info("Initialized with hallucination prevention")

        self.conversational_intelligence = ConversationalIntelligence()
        self.flow_manager = ConversationFlowManager()
        self.perspective_framework = PerspectiveAnalysisFramework()

        if api_key:
            self.bias_detector = BiasDetector(api_key)
            logger.info("Initialized bias mitigation framework")
        else:
            self.bias_detector = None
            logger.warning("No API key provided - bias mitigation unavailable")

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
        logger.info(f"Added reasoning step: {step_type} (confidence: {confidence})")
        return step
    
    def update_conversation_state(self, new_state: ConversationState):
        old_state = self.context.state
        self.context.state = new_state
        logger.info(f"State transition: {old_state.value} -> {new_state.value}")
    
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
    
    def handle_query(self, query: str, history: List) -> Optional[str]:
        # Processes query with boundary management

        # Semantic classification 
        classification = self.conversational_intelligence.classify_request(
            query, 
            history
        )
        
        # Record classification reasoning
        self.create_reasoning_step(
            "boundary_classification",
            classification['reasoning'],
            classification['confidence']
        )
        
        # Handle based on classification
        if not classification['in_scope']:
            # Generate graceful redirect
            redirect = self.conversational_intelligence.generate_graceful_redirect(
                query,
                classification
            )
            
            # Track flow
            self.flow_manager.update_flow_state(query, "redirect", False)
            
            # Add topic suggestions
            suggestions = self.flow_manager.suggest_next_topics("general")
            if redirect:
                redirect += "\n\nSome topics you might find interesting:\n"
                for s in suggestions:
                    redirect += f"- {s}\n"
            
            return redirect
        
        # Check for conversation breakdown
        if self.conversational_intelligence.detect_conversation_breakdown(history, query):
            repair = self.conversational_intelligence.repair_conversation(
                history,
                "confusion"
            )
            self.flow_manager.update_flow_state(query, "repair", True)
            return repair
        
        # Process normally if in scope
        return None
    
    def maintain_conversation_context(self, message: str, response: str):
        """Maintain conversation context and flow"""
        self.context.message_history.append((message, response))
        
        # Track topics
        if "election" in message.lower():
            self.flow_manager.topic_tracking["election"] = \
                self.flow_manager.topic_tracking.get("election", 0) + 1
        if "policy" in message.lower():
            self.flow_manager.topic_tracking["policy"] = \
                self.flow_manager.topic_tracking.get("policy", 0) + 1
        
        # Update conversation state
        if len(self.context.reasoning_chain) > 5:
            self.context.state = ConversationState.POLITICAL_DISCUSSION
    
    async def process_message_async(self, message: str, history: List[Tuple[str, str]]) -> str:
        # Completes async message processing with all components

        # Boundary management
        boundary_response = self.handle_query(message, history)
        if boundary_response:
            return boundary_response
        
        # Creates reasoning chain
        reasoning_step = self.create_reasoning_step(
            "query_analysis",
            f"Analyzing query: {message}",
            0.7
        )
        
        # Updates state
        self.update_conversation_state(ConversationState.POLITICAL_DISCUSSION)
        
        # Generates initial response
        initial_response = f"Providing balanced analysis of: {message}"
        
        # Applies bias mitigation
        if self.bias_detector:
            mitigated_response, biases = self.apply_bias_mitigation(initial_response)
            balanced_response = self.bias_detector.ensure_perspective_balance(message, mitigated_response)
        else:
            balanced_response = initial_response
        
        # Applies hallucination prevention
        validated_response = await self.validate_and_ground_response(message, balanced_response)
        
        # Updates conversation context
        self.maintain_conversation_context(message, validated_response)
        
        return validated_response
    
    def process_message(self, message: str, history: List[Tuple[str, str]]) -> str:
        # Synchronous wrapper for message processing
        if history is None:
            history = []
        return asyncio.run(self.process_message_async(message, history))


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