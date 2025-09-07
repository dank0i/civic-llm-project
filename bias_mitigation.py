# bias_mitigation.py
# 

import openai
import json
from typing import List, Dict, Tuple

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
        
        Finally, provide an analysis in JSON:
        {{
            "detected_biases": ["list of reasoned bias detections"],
            "corrections_made": ["specific corrections applied"],
            "neutralized_response": "bias-corrected version",
            "confidence_in_neutrality": float (0-1),
            "reasoning": "explanation of bias detection reasoning"
        }}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a bias detection system that uses reasoning, not keywords, in a political scenario, aiming for strict neutrality."
                    },
                    {"role": "user", "content": mitigation_prompt}
                ],
                temperature=0.2
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return analysis['neutralized_response'], analysis['detected_biases']
            
        except Exception as e:
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
        Finally, format as a JSON with 'analysis' and 'balanced_response'.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a perspective balance analyzer using systematic frameworks, and must remain politcally neutral."
                    },
                    {"role": "user", "content": balance_prompt}
                ],
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('balanced_response', initial_response)
            
        except Exception as e:
            return initial_response

