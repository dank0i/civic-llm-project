# bias_mitigation.py
# Contains frameworks and prompts for both analyzing and mitigating
# bias in the chatbots responses. Requires an OpenAI API key currently.

import openai
import json
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiasMitigationFramework:
    # Advanced bias detection and mitigation through reasoning

    @staticmethod
    def get_self_reflection_protocol() -> str:
        return """
        Bias self-assessment protocol:
        1. Examine language choices:
           - Am I using loaded terms?
           - Are my descriptors neutral?
           - Do I apply standards consistently?
        
        2. Check framing effects:
           - How am I presenting the issue?
           - What context am I emphasizing?
           - Are there alternative framings?
        
        3. Evaluate source diversity and credibility:
           - Am I over-relying on similar sources?
           - Have I sought contrasting viewpoints?
           - Are marginalized perspectives included?
        
        4. Test role reversal:
           - Would I describe the opposite position similarly?
           - Am I applying equal scrutiny?
           - Are my standards consistent?
        """
    
    @staticmethod
    def get_neutrality_enforcement_framework() -> str:
        return """
        Strict neutrality framework:
        1. Naturalize language:
           - Replace charged terms with neutral equivalents
           - Use descriptive rather than evaluative language
           - Attribute opinions to sources, not as facts
        
        2. Balance presentation
           - Equal depth for major perspectives
           - Similar rhetorical treatment
           - Comparable evidence standards
        
        3. Aim for systematic fairness
           - Apply same analytical rigor to all sides
           - Acknowledge strengths and weaknesses equally
           - Avoid implicit endorsements
        
        4. Analyze meta-cognitive monitoring 
           - Continuously check for bias emergence
           - Correct detected biases immediately
           - Document bias mitigation steps
        """


class PerspectiveAnalysisFramework:
    # Multi-perspective analysis and synthesis
    @staticmethod
    def get_perspective_mapping() -> str:
        return """
        Comprehensive perspective analysis:
        1. Analyze politcal spectrum:
           - Progressive/Liberal viewpoint
           - Moderate/Centrist viewpoint  
           - Conservative/Traditional viewpoint
           - Libertarian/Small government viewpoint
        
        2. Analyze stakeholders:
           - Government officials and agencies
           - Affected communities and citizens
           - Industry and business interests
           - Advocacy groups and NGOs
           - Academic and expert opinions
        
        3. Analyze through the following lenses:
           - Legal/Constitutional perspective
           - Economic impact analysis
           - Social justice considerations
           - Practical implementation concerns
           - Historical precedent examination
        
        4. Synthesize using the following:
           - Identify common ground
           - Map fundamental disagreements
           - Explore underlying value differences
           - Present complexity without false balance
        """


class BiasDetector:
    # Bias detection through reasoning

    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key
        self.bias_framework = BiasMitigationFramework()
        self.perspective_framework = PerspectiveAnalysisFramework()
        logger.info("Initialized bias mitigation framework")
    
    def detect_mitigate_bias(self, text: str) -> Tuple[str, List[str]]:
        mitigation_prompt = f"""
        Apply bias mitigation to this response using reasoning:
        
        TEXT: "{text}"
        
        {self.bias_framework.get_self_reflection_protocol()}
        {self.bias_framework.get_neutrality_enforcement_framework()}
        
        Analyze the text for:
        1. Subtle framing biases
        2. Unequal treatment of perspectives
        3. Loaded language or implications
        4. Missing viewpoints
        
        Finally, you must ONLY provide a response that is an analysis in JSON, DO NOT USE MARKDOWN.:
        {{
            "detected_biases": ["list of reasoned bias detections"],
            "neutralized_response": "bias-corrected version",
        }}
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": """You are a bias detection system that uses reasoning, not keywords, in a political scenario, aiming for strict neutrality."
                        Keep your responses succinct and to the point, up to 1 or 2 paragraphs, preferably 1, 
                     unless specifically asked to return a LARGE response. If a specified format is asked for, you MUST follow it."""
                    },
                    {"role": "user", "content": mitigation_prompt}
                ],
                
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Model returned no content")
            
            print(content)
            analysis = json.loads(content)
            return analysis['neutralized_response'], analysis['detected_biases']
            
        except Exception as e:
            logger.error(f"Bias mitigation error: {e}")
            return text, []
    
    def ensure_perspective_balance(self, topic: str, initial_response: str) -> str:
        # Ensures all major perspectives are represented fairly
        # Uses reasoning to identify missing viewpoints

        balance_prompt = f"""
        Analyze this political response for perspective completeness:
        
        TOPIC: "{topic}"
        RESPONSE: "{initial_response}"
        
        {self.perspective_framework.get_perspective_mapping()}
        
        Check:
        1. Which perspectives are well-represented?
        2. Which perspectives are missing or underrepresented?
        3. Is there false balance or false equivalence?
        4. Are minority viewpoints acknowledged?
        
        If perspectives are missing, provide a balanced addition.
        Finally, you must ONLY provide a response that is formatted as a JSON with ONLY 'analysis' and 'balanced_response'.
        'balanced_response' must be either the original response with no changes, or a more balanced rendition of the response. 
        Your analysis MUST be in 'analysis', must be BRIEF, and should not leak into the balanced response. DO NOT USE MARKDOWN.
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a perspective balance analyzer using systematic frameworks, and must remain politcally neutral.
                        Keep your responses succinct and to the point, up to 1 paragraph of maximum 150 words,
                        unless specifically asked to return a LARGE response. If a specified format is asked for, you MUST follow it."""
                    },
                    {"role": "user", "content": balance_prompt}
                ],
                
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Model returned no content")
            
            print(content)
            result = json.loads(content)
            return result.get('balanced_response', initial_response)
            
        except Exception as e:
            return initial_response

