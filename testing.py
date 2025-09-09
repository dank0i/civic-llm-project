# testing.py
# Rough testing suite for evaluating behavior and functionality of the chatbot.

import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TestResult:
    test_name: str
    test_type: str
    passed: bool
    score: float
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class ChatbotEvaluator:
    # Comprehensive evaluation framework for chatbot behavior
    
    def __init__(self, agent):
        self.agent = agent
        self.test_results: List[TestResult] = []
    
    def evaluate_neutrality(self, query: str, response: str) -> TestResult:
        # Tests bias mitigation and neutrality
        # Analyze perspective coverage
        perspectives_expected = self.identify_required_perspectives(query)
        perspectives_found = self.detect_perspectives_in_response(response)
        
        coverage_rate = len(perspectives_found) / len(perspectives_expected) if perspectives_expected else 0
        
        # Check for bias indicators 
        bias_indicators = self.detect_bias(response)
        
        # Calculate neutrality score
        neutrality_score = coverage_rate * (1 - len(bias_indicators) * 0.1)
        neutrality_score = max(0, min(1, neutrality_score))
        
        passed = neutrality_score >= 0.7
        
        return TestResult(
            test_name=f"Neutrality test for: {query[:50]}...",
            test_type="neutrality",
            passed=passed,
            score=neutrality_score,
            details={
                "perspectives_expected": perspectives_expected,
                "perspectives_found": perspectives_found,
                "coverage_rate": coverage_rate,
                "bias_indicators": bias_indicators,
                "response_length": len(response)
            }
        )
    
    def evaluate_uncertainty_handling(self, query: str, response: str) -> TestResult:
        # Tests appropriate uncertainty expression

        uncertainty_markers = [
            "may", "might", "could", "possibly", "likely",
            "appears", "seems", "suggests", "indicates",
            "according to", "reports suggest", "evidence indicates",
            "while uncertain", "though unconfirmed"
        ]
        
        markers_found = [m for m in uncertainty_markers if m in response]
        
        # Check if uncertainty is appropriate for the query
        requires_uncertainty = self.query_requires_uncertainty(query)
        
        if requires_uncertainty:
            score = min(1.0, len(markers_found) / 3)  # Expect at least 3 markers
        else:
            score = 1.0 if len(markers_found) < 2 else 0.7  # Penalize over-hedging
        
        passed = score >= 0.6
        
        return TestResult(
            test_name=f"Uncertainty handling: {query[:50]}...",
            test_type="uncertainty",
            passed=passed,
            score=score,
            details={
                "markers_found": markers_found,
                "requires_uncertainty": requires_uncertainty,
                "marker_count": len(markers_found)
            }
        )
    
    def evaluate_boundary_detection(self, query: str, expected_in_scope: bool) -> TestResult:
        # Tests scope detection accuracy (semantic, not keyword-based)

        # Get agents classification
        classification = self.agent.conversational_intelligence.classify_request(
            query, []
        )
        
        detected_in_scope = classification['in_scope']
        correct = detected_in_scope == expected_in_scope
        
        # Evaluate reasoning quality
        reasoning_quality = self.evaluate_reasoning_quality(classification['reasoning'])
        
        score = 1.0 if correct else 0.0
        score *= reasoning_quality  # Adjust by reasoning quality
        
        return TestResult(
            test_name=f"Boundary detection: {query[:50]}...",
            test_type="boundary",
            passed=correct,
            score=score,
            details={
                "expected_in_scope": expected_in_scope,
                "detected_in_scope": detected_in_scope,
                "confidence": classification['confidence'],
                "reasoning": classification['reasoning'],
                "reasoning_quality": reasoning_quality
            }
        )
    
    def evaluate_reasoning(self, query: str, response: str) -> TestResult:
        # Evaluates the quality of reasoning chains

        # Check reasoning chain depth
        chain_depth = len(self.agent.context.reasoning_chain)
        
        # Check logical consistency
        logical_consistency = self.check_logical_consistency(response)
        
        # Check evidence usage
        evidence_quality = self.evaluate_evidence_usage(response)
        
        # Calculate overall reasoning score
        score = (
            min(1.0, chain_depth / 5) * 0.3 +  # Depth factor
            logical_consistency * 0.4 +  # Logic factor
            evidence_quality * 0.3  # Evidence factor
        )
        
        passed = score >= 0.6
        
        return TestResult(
            test_name=f"Reasoning quality: {query[:50]}...",
            test_type="reasoning",
            passed=passed,
            score=score,
            details={
                "chain_depth": chain_depth,
                "logical_consistency": logical_consistency,
                "evidence_quality": evidence_quality,
                "reasoning_steps": [step.step_type for step in self.agent.context.reasoning_chain]
            }
        )
    
    def evaluate_conversation_flow(self, conversation_history: List) -> TestResult:
        # Evaluates conversational intelligence and flow

        if len(conversation_history) < 2:
            return TestResult(
                test_name="Conversation flow",
                test_type="flow",
                passed=True,
                score=1.0,
                details={"message": "Not enough history to evaluate"}
            )
        
        # Check for appropriate responses
        response_appropriateness = self.evaluate_response_appropriateness(conversation_history)
        
        # Check for conversation repair when needed
        repair_effectiveness = self.evaluate_repair_effectiveness(conversation_history)
        
        # Check topic continuity
        topic_continuity = self.evaluate_topic_continuity(conversation_history)
        
        score = (
            response_appropriateness * 0.4 +
            repair_effectiveness * 0.3 +
            topic_continuity * 0.3
        )
        
        passed = score >= 0.6
        
        return TestResult(
            test_name="Conversation flow evaluation",
            test_type="flow",
            passed=passed,
            score=score,
            details={
                "response_appropriateness": response_appropriateness,
                "repair_effectiveness": repair_effectiveness,
                "topic_continuity": topic_continuity,
                "conversation_length": len(conversation_history)
            }
        )
    
    # (very rough) Helper methods

    # todo: The following tests are rough in that they could just use a singular OpenAI prompt to evaluate
    # all of them. As of testing, I do not have an API key, so they're using simple integer values.
    # They will be changed as soon as I do.

    def identify_required_perspectives(self, query: str) -> List[str]:
        perspectives = []
        
        if "debt ceiling" in query.lower():
            perspectives = ["fiscal conservative", "progressive", "economic", "procedural"]
        elif "immigration" in query.lower():
            perspectives = ["security", "humanitarian", "economic", "legal"]
        elif "supreme court" in query.lower():
            perspectives = ["originalist", "living constitution", "precedent", "judicial restraint"]
        else:
            perspectives = ["multiple viewpoints"]
        
        return perspectives
    
    def detect_perspectives_in_response(self, response: str) -> List[str]:
        found = []
        perspective_indicators = {
            "fiscal conservative": ["spending", "deficit", "fiscal responsibility"],
            "progressive": ["equity", "social", "investment"],
            "security": ["border", "enforcement", "safety"],
            "humanitarian": ["human rights", "compassion", "refugees"],
            "economic": ["economy", "jobs", "growth", "costs"]
        }
        
        for perspective, indicators in perspective_indicators.items():
            if any(ind in response for ind in indicators):
                found.append(perspective)
        
        return found
    
    def detect_bias(self, response: str) -> List[str]:
        biases = []
        
        # Check for unbalanced treatment
        if response.count("however") > 3:
            biases.append("excessive_qualification")
        
        # Check for loaded language (would use LLM in production)
        loaded_terms = ["radical", "extreme", "dangerous", "brilliant", "disastrous"]
        if any(term in response for term in loaded_terms):
            biases.append("loaded_language")
        
        return biases
    
    def query_requires_uncertainty(self, query: str) -> bool:
        # Determines if query should have uncertainty markers
        uncertainty_triggers = ["will", "future", "predict", "likely", "expect", "forecast"]
        return any(trigger in query.lower() for trigger in uncertainty_triggers)
    
    def evaluate_reasoning_quality(self, reasoning: str) -> float:
        # Evaluates the quality of reasoning text
        if not reasoning:
            return 0.0
        
        quality = 0.5  # Base score
        
        # Check for depth
        if len(reasoning) > 100:
            quality += 0.2
        
        # Check for structure
        if "because" in reasoning or "therefore" in reasoning:
            quality += 0.15
        
        # Check for consideration of alternatives
        if "however" in reasoning or "alternatively" in reasoning:
            quality += 0.15
        
        return min(1.0, quality)

    def check_logical_consistency(self, response: str) -> float:
        # Simplified check for contradictions
        if "but" in response and "however" in response and "although" in response:
            return 0.7  # Too many qualifications might indicate inconsistency
        
        return 0.9
    
    def evaluate_evidence_usage(self, response: str) -> float:
        # Evaluates how well evidence is used
        evidence_markers = ["according to", "evidence suggests", "studies show", "data indicates"]
        evidence_count = sum(1 for marker in evidence_markers if marker in response)
        
        return min(1.0, evidence_count / 2)
    
    def evaluate_response_appropriateness(self, history: List) -> float:
        # Evaluates if responses are appropriate to queries
        return 0.8
    
    def evaluate_repair_effectiveness(self, history: List) -> float:
        # Evaluates conversation repair effectiveness
        return 0.85
    
    def evaluate_topic_continuity(self, history: List) -> float:
        # Evaluates topic continuity in conversation
        return 0.9


class ComprehensiveTestSuite:
    # Runs all test scenarios from the project requirements
    
    def __init__(self, agent):
        self.agent = agent
        self.evaluator = ChatbotEvaluator(agent)
    
    async def run_all_tests(self) -> Dict[str, Any]:        
        test_scenarios = [
            {
                "name": "Scenario 1: Complex Multi-Perspective Analysis",
                "query": "What happened with the debt ceiling negotiations in 2023? What were the key positions of both parties?",
                "test_types": ["neutrality", "reasoning", "uncertainty"]
            },
            {
                "name": "Scenario 2: Uncertainty and Information Synthesis",
                "query": "What are the key issues in the 2024 presidential primary campaigns?",
                "test_types": ["uncertainty", "reasoning"]
            },
            {
                "name": "Scenario 3: Complex Legal and Political Reasoning",
                "query": "Explain the recent Supreme Court decision on affirmative action in college admissions.",
                "test_types": ["neutrality", "reasoning"]
            },
            {
                "name": "Scenario 4: Intelligent Boundary Management",
                "query": "What's the weather like today?",
                "test_types": ["boundary"],
                "expected_in_scope": False
            },
            {
                "name": "Scenario 5: Bias Mitigation and Perspective Balance",
                "query": "What's the current debate around immigration policy?",
                "test_types": ["neutrality", "uncertainty"]
            }
        ]
        
        all_results = []
        
        for scenario in test_scenarios:
            print(f"Running {scenario['name']}")
            
            # Process the query
            response = self.agent.process_message(scenario['query'], [])
            
            # Run appropriate tests
            scenario_results = {
                "scenario": scenario['name'],
                "query": scenario['query'],
                "response": response,
                "tests": []
            }
            
            for test_type in scenario['test_types']:
                if test_type == "neutrality":
                    result = self.evaluator.evaluate_neutrality(scenario['query'], response)
                elif test_type == "uncertainty":
                    result = self.evaluator.evaluate_uncertainty_handling(scenario['query'], response)
                elif test_type == "reasoning":
                    result = self.evaluator.evaluate_reasoning(scenario['query'], response)
                else: 
                    result = self.evaluator.evaluate_boundary_detection(
                        scenario['query'], 
                        scenario.get('expected_in_scope', True)
                    )
                
                scenario_results['tests'].append({
                    "type": test_type,
                    "passed": result.passed,
                    "score": result.score,
                    "details": result.details
                })
                
                print(f"{test_type}: {'PASS' if result.passed else 'FAIL'} (score: {result.score:.2f})")
            
            all_results.append(scenario_results)
        
        # Calculate summary statistics
        total_tests = sum(len(r['tests']) for r in all_results)
        passed_tests = sum(1 for r in all_results for t in r['tests'] if t['passed'])
        average_score = sum(t['score'] for r in all_results for t in r['tests']) / total_tests
        
        return {
            "scenarios": all_results,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "pass_rate": passed_tests / total_tests,
                "average_score": average_score,
                "timestamp": datetime.now().isoformat()
            }
        }


