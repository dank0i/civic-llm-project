# hallucination_prevention.py
# Contains the hallucination prevention architecture for the chatbot, 
# using reasoning with both Tavily API and OpenAI to ensure information quality.

import aiohttp
import json
import openai
from typing import Any, Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InformationValidator:
    # Reasoning-based information validation and hallucination prevention
    
    def __init__(self, tavily_api_key: Optional[str] = None):
        self.tavily_api_key = tavily_api_key
        
    async def search_information(self, query: str) -> Dict[str, Any]:
        # Information grounding through Tavily API
        if not self.tavily_api_key:
            return {
                "success": False,
                "message": "Search unavailable - using reasoning only",
                "fallback": True
            }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.tavily_api_key,
                        "query": query,
                        "search_depth": "advanced",
                        "max_results": 5
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "results": data.get("results", []),
                            "answer": data.get("answer", "")
                        }
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {
                "success": False,
                "fallback": True
            }
        
        return {"success": False, "fallback": True}
    
    def validate_information_coherence(self, claim: str, context: str) -> Dict[str, Any]:
        validation_prompt = f"""
        Assess the coherence and plausibility of this claim through reasoning:
        
        CLAIM: "{claim}"
        CONTEXT: "{context}"
        
        Apply these reasoning steps:
        
        1. Logical consistency:
           - Does the claim contradict itself?
           - Are there logical gaps or leaps?
           - Do the parts form a coherent whole?
        
        2. Process practicality
           - Does this align with how political processes work?
           - Are the described mechanisms realistic?
           - Would this require impossible coordination?
        
        3. Timeline coherence
           - Do the timelines make sense?
           - Are cause-effect relationships temporally valid?
           - Are there anachronistic elements?
        
        4. Stakeholder behavior:
           - Would actors behave as described?
           - Are motivations consistent with interests?
           - Do reactions match expected patterns?
        
        Finally, you must ONLY provide a response that is an assessment as JSON:
        {{
            "coherence_score": float (0-1),
            "logical_issues": ["list of logical problems"],
            "plausibility_assessment": "detailed reasoning",
            "confidence_level": float (0-1),
            "uncertainty_factors": ["sources of uncertainty"]
        }}
        """
        
        # This would normally call the LLM
        # For now, return a structured response
        return {
            "coherence_score": 0.7,
            "logical_issues": [],
            "plausibility_assessment": "Requires reasoning engine",
            "confidence_level": 0.6,
            "uncertainty_factors": ["Limited context", "Evolving situation"]
        }


class UncertaintyCalibrator:
    # Quantifies and expresses uncertainty
    
    @staticmethod
    def calculate_confidence(
        source_quality: str,
        temporal_relevance: int,  # days old
        consensus_level: str
    ) -> float:
        # Base confidence from source quality
        source_scores = {
            "primary": 0.9,
            "secondary": 0.7,
            "analysis": 0.5,
            "speculation": 0.3,
            "unknown": 0.2
        }
        base_confidence = source_scores.get(source_quality, 0.5)
        
        # Temporal degradation
        if temporal_relevance < 7:
            temporal_factor = 0
        elif temporal_relevance < 30:
            temporal_factor = -0.1
        elif temporal_relevance < 180:
            temporal_factor = -0.2
        else:
            temporal_factor = -0.3
        
        # Consensus adjustment
        consensus_adjustments = {
            "universal": 0.2,
            "broad": 0.1,
            "divided": -0.1,
            "controversial": -0.2,
            "unknown": 0
        }
        consensus_factor = consensus_adjustments.get(consensus_level, 0)
        
        # Calculate final confidence
        final_confidence = max(0, min(1, base_confidence + temporal_factor + consensus_factor))
        
        return final_confidence
    
    @staticmethod
    def express_uncertainty(confidence: float, statement: str) -> str:
        # Adds appropriate uncertainty markers based on confidence
        if confidence > 0.8:
            prefix = "Based on verified sources, "
            suffix = ""
        elif confidence > 0.6:
            prefix = "Available evidence suggests that "
            suffix = ""
        elif confidence > 0.4:
            prefix = "While there is some uncertainty, "
            suffix = " (though this should be verified)"
        elif confidence > 0.2:
            prefix = "With limited confidence, it appears that "
            suffix = " (this remains largely unconfirmed)"
        else:
            prefix = "I cannot reliably confirm this, but "
            suffix = " (please treat this as speculative)"
        
        return f"{prefix}{statement}{suffix}"


class HallucinationPreventer:
    # Main hallucination prevention system using reasoning chains
    
    def __init__(self, api_key: Optional[str] = None, tavily_key: Optional[str] = None):
        self.api_key = api_key
        self.validator = InformationValidator(tavily_key)
        self.calibrator = UncertaintyCalibrator()
        
        if api_key:
            openai.api_key = api_key
    
    def reason_about_information_gaps(self, query: str, available_info: Dict) -> Dict[str, Any]:
        # Identifies what we know, don't know, and can't know through reasoning

        if not self.api_key:
            return {
                "known_facts": [],
                "unknown_aspects": ["API key required for reasoning"],
                "unknowable_aspects": [],
                "confidence_map": {}
            }
        
        reasoning_prompt = f"""
        Analyze information completeness for: "{query}"
        
        Available information: {json.dumps(available_info, indent=2)}
        
        Categorize our knowledge:
        
        1. Known facts (high confidence):
           - What can we state with confidence?
           - What is well-established?
        
        2. Unknown, but knowable/practical:
           - What information gaps exist?
           - What could be researched further?
        
        3. Inherently uncertain, due to various factors:
           - What involves future predictions?
           - What depends on private information?
           - What involves subjective interpretation?
        
        4. Map confidence:
           - Assign confidence to each claim
           - Identify sources of uncertainty
        
        You must ONLY provide a response that is a JSON with these categories.
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an information completeness analyzer, assigned to determine confidence scores for sources and facts."
                    },
                    {"role": "user", "content": reasoning_prompt}
                ],
                temperature=0.3
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Model returned no content")
            
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Information gap analysis error: {e}")
            return {
                "known_facts": [],
                "unknown_aspects": ["Analysis failed"],
                "unknowable_aspects": [],
                "confidence_map": {}
            }
    
    def prevent_hallucination(
        self, 
        query: str, 
        initial_response: str
    ) -> str:
        # Step 1: Identify claims in the response
        claims = self.extract_claims(initial_response)
        
        # Step 2: Assess each claim's confidence
        validated_response = initial_response
        for claim in claims:
            confidence = self.assess_claim_confidence(claim, query)
            if confidence < 0.5:
                # Add uncertainty markers
                validated_response = self.add_uncertainty_to_claim(
                    validated_response, 
                    claim, 
                    confidence
                )
        
        return validated_response
    
    def extract_claims(self, text: str) -> List[str]:
        # Extract claims through reasoning about what constitutes a factual claim

        if not self.api_key:
            # fallback - just split sentences
            sentences = text.split('. ')
            return [s for s in sentences if len(s) > 20]
        
        # Use LLM reasoning to identify claims
        extraction_prompt = f"""
        Analyze this text and identify factual claims using reasoning:
        
        Text: "{text}"
        
        Apply this reasoning framework:
        1. Is this statement asserting something as fact?
        2. Could this statement be verified or falsified?
        3. Does it make a specific claim about reality?
        
        Do NOT use keyword matching. Use semantic understanding.
        
        Finally, you must ONLY provide a response that is JSON with just the claim texts:
        {{
            "claims": ["claim 1 text", "claim 2 text", ...]
        }}
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a claim identifier using reasoning, not pattern matching."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.3
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Model returned no content")
        
            result = json.loads(content)
            return result.get("claims", [])
        except:
            sentences = text.split('. ')
            return [s for s in sentences if len(s) > 20]

    
    def assess_claim_confidence(self, claim: str, context: str) -> float:
        # Assess confidence through reasoning about the claim's nature

        if not self.api_key:
            # fallback - return moderate confidence
            return 0.5
        
        # Use reasoning chains to assess confidence
        assessment_prompt = f"""
        Assess confidence in this claim through reasoning:
        
        Claim: "{claim}"
        Context: "{context}"
        
        Apply this reasoning chain:
        
        1. Claim specificity reasoning:
        - Does the claim make specific, verifiable assertions?
        - Are there concrete details or vague generalizations?
        
        2. Epistemic reasoning:
        - What kind of knowledge would be needed to verify this?
        - Is this knowable in principle?
        
        3. Contextual coherence:
        - Does this claim fit logically with the context?
        - How plausible is this given what we know?
        
        4. Temporal reasoning:
        - Is this about past (more certain) or future (less certain)?
        
        You must ONLY provide a response that is just a confidence score between 0 and 1 as a number.
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Return only a confidence score between 0 and 1."},
                    {"role": "user", "content": assessment_prompt}
                ],
                temperature=0.2
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Model returned no content")
            
            # Parse the response as a float
            confidence_str = content.strip()
            return float(confidence_str)
        except:
            return 0.5  # Default moderate confidence
    
    def add_uncertainty_to_claim(self, text: str, claim: str, confidence: float) -> str:
        # Adds uncertainty markers to a specific claim
        uncertainty_prefix = self.calibrator.express_uncertainty(confidence, "")
        
        # Find and modify the claim in the text
        if claim in text:
            modified_claim = f"{uncertainty_prefix}{claim}"
            text = text.replace(claim, modified_claim)
        
        return text

