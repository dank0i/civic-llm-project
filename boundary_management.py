# boundary_management.py
# Contains the boundary management framework for the chatbot, including reasoning
# methods / prompts and graceful topic redirection.

from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import openai
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BoundaryManagementFramework:
    # Intelligent scope management through reasoning
    
    @staticmethod
    def get_scope_reasoning_framework() -> str:
        return """
        Intelligent scope management:
        1. Request classification reasoning:
           - Analyze semantic content, not keywords
           - Consider context and intent
           - Evaluate relevance to political events
        
        2. Boundary decision framework:
           - Is this about political events/policies?
           - Does it require political analysis?
           - Can I provide value within scope?
        
        3. Graceful redirection:
           - Acknowledge the request respectfully
           - Explain reasoning for scope limitation
           - Offer relevant alternatives if possible
           - Maintain helpful conversational tone
        
        4. Edge case handling:
           - Political history: Generally in scope
           - Political theory: Case-by-case evaluation
           - Current events with political implications: In scope
           - Personal political advice: Explain limitations
        """
    
    @staticmethod
    def get_conversation_repair_framework() -> str:
        return """
        Conversation repair mechanism:
        1. Misunderstanding detection:
           - Monitor for confusion signals
           - Check response relevance
           - Identify conversation breakdowns
        
        2. Clarification strategies:
           - Ask focused follow-up questions
           - Provide examples of what you can discuss
           - Reframe the conversation productively
        
        3. Context recovery:
           - Reference earlier conversation points
           - Rebuild shared understanding
           - Re-establish conversation goals
        
        4. Productive redirection:
           - Guide toward answerable questions
           - Suggest related topics in scope
           - Maintain engagement while respecting boundaries
        """


class ConversationalIntelligence:
    # Conversational flow management
    
    def __init__(self, api_key: Optional[str] = None):
        self.boundary_framework = BoundaryManagementFramework()
        self.api_key = api_key
    
    def classify_request(self, query: str, conversation_history: List) -> Dict:
        # Classifies query through reasoning about intent and meaning

        if not hasattr(self, 'api_key') or not self.api_key:
            # fallback - return basic classification
            return {
                "in_scope": True,  # Default to attempting help
                "confidence": 0.5,
                "reasoning": "Cannot perform deep semantic analysis without LLM",
                "scope_type": "edge_case",
                "suggested_response_type": "clarification"
            }
        
        # Use multi-step reasoning to understand the query
        classification_prompt = f"""
        Classify this query using semantic reasoning about meaning and intent.

        Query: "{query}"
        Recent context: {conversation_history[-2:] if conversation_history else "None"}
        
        Apply this reasoning framework:
        
        1. Analyze intent:
        - What is the user trying to accomplish?
        - What kind of response would satisfy their need?
        
        2. Reason about domain:
        - Does this require political knowledge or analysis?
        - Is this about governance, policy, or political events?
        
        3. Capability matching:
        - Can we provide value while maintaining neutrality?
        - Is this within our political analysis scope?
        
        4. Semantic categorization:
        - Based on meaning, what category is this?
        
        You must ONLY provide a JSON response that is returned as the following format. DO NOT USE MARKDOWN.
        Your reasoning should be succinct (and somewhat brief - 150 words max):
        {{
            "in_scope": boolean,
            "confidence": float (0-1),
            "reasoning": "semantic reasoning explanation",
            "scope_type": "political_event|political_analysis|edge_case|out_of_scope",
            "suggested_response_type": "full_analysis|redirect|clarification"
        }}
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """You are a semantic understanding system that reasons about meaning."
                    Keep your responses brief, succinct and to the point, up to 1 paragraph of maximum 150 words,
                     unless specifically asked to return a LARGE response. If a specified format is asked for, you MUST follow it."""},
                    {"role": "user", "content": classification_prompt}
                ],
                
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Model returned no content")
            
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Semantic classification error: {e}")
            return {
                "in_scope": True,
                "confidence": 0.5,
                "reasoning": "Classification failed, defaulting to attempt help",
                "scope_type": "edge_case",
                "suggested_response_type": "clarification"
            }
    
    def generate_graceful_redirect(self, query: str, classification: Dict) -> str:
        # Generates intelligent redirects to maintain conversation flow
        if classification['scope_type'] == 'out_of_scope':
            redirect_template = f"""
                I understand you're asking about "{query}". 

                Based on my analysis, this appears to be outside my specific scope of political events and policy discussions. 
                {classification['reasoning']}

                I'm designed to provide balanced, multi-perspective analysis of political events, policies, and related topics. 
                I'd be happy to discuss:
                - Recent political developments and their implications
                - Policy debates and different stakeholder perspectives  
                - Political history and its relevance to current events
                - Constitutional and legal aspects of political issues

                Is there a political topic or event you'd like to explore?"""
            
        elif classification['scope_type'] == 'edge_case':
            redirect_template = f"""
                Your question about "{query}" touches on areas that may be partially within my scope.

                {classification['reasoning']}

                I can provide political context and analysis for aspects of this topic. Would you like me to focus on:
                - The political dimensions of this issue?
                - Related policy debates or governmental responses?
                - How different political perspectives view this topic?

                What aspect would be most helpful for you?"""
                    
        else:
            redirect_template = ""
        
        return redirect_template
    
    def detect_conversation_breakdown(self, history: List, current_query: str) -> bool:
        # Detects when conversation is going off track
        if len(history) < 2:
            return False
        
        # Check for repetitive patterns
        if history:
            recent_messages = [msg for msg, _ in history[-3:]] if len(history) >= 3 else []
            if len(set(recent_messages)) == 1 and len(recent_messages) > 1:
                return True
        
        # Check for confusion signals
        confusion_signals = [
            "don't understand",
            "that's not what I asked",
            "can you just",
            "why can't you",
            "this is frustrating"
        ]
        
        return any(signal in current_query.lower() for signal in confusion_signals)
    
    def repair_conversation(self, history: List, issue_type: str) -> str:
        # Generates conversation repair responses
        
        if issue_type == "confusion":
            return """
                I sense there might be some confusion. Let me clarify what I can help with:

                I'm specialized in providing balanced analysis of political events and policies. I can:
                - Explain recent political developments from multiple perspectives
                - Analyze policy proposals and their potential impacts
                - Discuss political history and constitutional matters
                - Present different viewpoints on controversial political topics

                Could you rephrase your question or let me know which political topic interests you?"""
        
        elif issue_type == "repetition":
            return """
                I notice we might be going in circles. Let me try a different approach.

                What specific aspect of political events or policy would you like to explore? 
                For example:
                - A recent political development you'd like understood
                - A policy debate you want analyzed from different angles
                - A political process or system you'd like explained

                How can I best help you with political analysis today?"""
        
        return "Let me help you with political analysis. What would you like to know?"


class ConversationFlowManager:
    # Manages overall conversation flow and state transitions
    
    def __init__(self):
        self.state_history = []
        self.topic_tracking = {}
        self.engagement_score = 1.0
    
    def update_flow_state(self, query: str, response_type: str, success: bool):
        # Track conversation flow for optimization
        self.state_history.append({
            "query": query,
            "response_type": response_type,
            "success": success,
            "timestamp": datetime.now()
        })
        
        # Update engagement score
        if success:
            self.engagement_score = min(1.0, self.engagement_score + 0.1)
        else:
            self.engagement_score = max(0.0, self.engagement_score - 0.2)
    
    def suggest_next_topics(self, current_topic: str) -> List[str]:
        # Suggest related political topics to maintain engagement and maintain graceful flow
        topic_relations = {
            "election": [
                "How do different voting systems work?",
                "What are the key policy differences between parties?",
                "How has campaign finance evolved?"
            ],
            "policy": [
                "What are the main arguments for and against this policy?",
                "How do similar policies work in other countries?",
                "What are the implementation challenges?"
            ],
            "supreme_court": [
                "How does the judicial nomination process work?",
                "What are the different judicial philosophies?",
                "How do court decisions impact policy?"
            ]
        }
        
        # Find related topics
        for key, suggestions in topic_relations.items():
            if key in current_topic.lower():
                return suggestions
        
        # Default suggestions
        return [
            "What recent political development would you like to understand?",
            "Any policy debate you'd like analyzed from multiple angles?",
            "Questions about how political processes work?"
        ]
    
    def get_engagement_metrics(self) -> Dict[str, Any]:
        # Get conversation engagement metrics, used to evaluate flow 
        return {
            "engagement_score": self.engagement_score,
            "total_interactions": len(self.state_history),
            "success_rate": sum(1 for s in self.state_history if s['success']) / len(self.state_history) if self.state_history else 0,
            "topics_discussed": list(self.topic_tracking.keys())
        }
