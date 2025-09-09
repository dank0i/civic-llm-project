# main.py
# The actual chatbot and Gradio interface.

from agent import PoliticalChatbotAgent
from testing import ComprehensiveTestSuite
import gradio as gr
from gradio import themes 
import asyncio
from typing import List, Optional, Tuple
import json
import os

def create_gradio_interface(api_key: str, tavily_key: Optional[str] = None):
    # Initialize the complete chatbot
    chatbot = PoliticalChatbotAgent(api_key, tavily_key)
    
    # Define the chat function
    async def chat_function(message: str, history: List[Tuple[str, str]]):
        # Processes chat messages asynchronously
        try:
            response = await chatbot.process_message(message, history)
            return response
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return "I apologize, but I encountered an error processing your request. Please try rephrasing your question."
    
    def sync_chat(message: str, history: List[Tuple[str, str]]):
        # Synchronous wrapper for Gradio
        return asyncio.run(chat_function(message, history))
    
    # Creates custom (but barebones) CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Inter', sans-serif;
    }
    .chat-wrap {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    """
    
    # Creates the interface
    with gr.Blocks(css=custom_css, theme=
                   themes.Soft()) as demo:
        gr.Markdown(
            """
            # Political Events AI Chatbot
            
            **Advanced AI for balanced political discussion and analysis**
            
            This chatbot provides:
            - Multi-perspective analysis of political events
            - Strict political neutrality through sophisticated bias mitigation
            - Intelligent reasoning chains for complex topics
            - Appropriate uncertainty expression
            - Semantic understanding

            Responses may take up to 30-45 seconds, depending on the specificity of the prompt,
            to ensure information quality and bias mitigation.

            Simply enter your message below to start chatting.
            
            ---
            """
        )
        
        # Creates chat interface
        chatbot_ui = gr.Chatbot(
            label="Political Discussion",
            height=500,
            bubble_full_width=False
        )
        
        msg = gr.Textbox(
            label="Your Question",
            placeholder="Ask about political events, policies, or governmental processes...",
            lines=2
        )
        
        with gr.Row():
            submit = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")
        
        # Example queries
        gr.Examples(
            examples=[
                "What happened with the debt ceiling negotiations in 2023? What were the key positions of both parties?",
                "What are the key issues in the 2024 presidential primary campaigns?",
                "Explain the recent Supreme Court decision on affirmative action in college admissions.",
                "What's the current debate around immigration policy?",
                "How does the Federal Reserve's interest rate policy work?",
                "What's the weather like today?",  # Tests boundary detection
            ],
            inputs=msg,
            label="Example Questions"
        )
        
        # Adds information panel
        with gr.Accordion("How to Use", open=False):
            gr.Markdown(
                """
                ### Getting the Best Results
                
                1. **Be specific** - Include context and timeframes when relevant
                2. **Ask for perspectives** - Request multiple viewpoints on controversial topics
                3. **Complex topics welcome** - The system handles multi-layered political analysis
                4. **Follow-up questions** - Build on previous responses for deeper understanding
                
                ### What This Chatbot Does Well
                
                - Explains political events from multiple perspectives
                - Analyzes policy proposals and their implications
                - Discusses constitutional and legal matters
                - Provides historical political context
                
                ### Limitations
                
                - Not for personal political advice
                - Cannot predict future events with certainty
                - May not have information on very recent events
                - Maintains strict political neutrality
                """
            )
        
        # Adds debugging panel (hidden by default)
        with gr.Accordion("🔧 Debug Information", open=False):
            debug_output = gr.JSON(label="Internal State")
            
            def get_debug_info():
                return {
                    "conversation_state": chatbot.context.state.value,
                    "reasoning_chain_length": len(chatbot.context.reasoning_chain),
                    "detected_biases": chatbot.context.detected_biases[-5:] if chatbot.context.detected_biases else [],
                    "confidence_level": chatbot.context.confidence_level,
                    "engagement_score": chatbot.flow_manager.engagement_score
                }
            
            refresh_debug = gr.Button("Refresh Debug Info")
            refresh_debug.click(get_debug_info, outputs=debug_output)
        
        # Sets up event handlers
        def respond(message, chat_history):
            bot_message = sync_chat(message, chat_history)
            chat_history.append((message, bot_message))
            return "", chat_history
        
        msg.submit(respond, [msg, chatbot_ui], [msg, chatbot_ui])
        submit.click(respond, [msg, chatbot_ui], [msg, chatbot_ui])
        clear.click(lambda: None, None, chatbot_ui, queue=False)
    
    return demo

def run_tests(api_key: str, tavily_key: Optional[str] = None):
    # Initialize chatbot
    chatbot = PoliticalChatbotAgent(api_key, tavily_key)
    
    # Initialize test suite
    test_suite = ComprehensiveTestSuite(chatbot)
    
    # Run all tests
    results = asyncio.run(test_suite.run_all_tests())
    
    # Save results to file
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Test results saved to test_results.json")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Political Events AI Chatbot")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--tavily-key", type=str, help="Tavily API key (optional)")
    
    args = parser.parse_args()
    
    # Get API keys
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    tavily_key = args.tavily_key or os.getenv("TAVILY_API_KEY")
    
    if not api_key:
        print("OpenAI API key required")
        print("Set OPENAI_API_KEY environment variable or use --api-key flag")
        return
    
    print("Political Events AI Chatbot")
    if args.test:
        # Run tests
        run_tests(api_key, tavily_key)
    else:
        # Launch web interface
        print("Launching web interface...")
        if not tavily_key:
            print("Note: Running without Tavily API (search functionality limited)")
        
        demo = create_gradio_interface(api_key, tavily_key)
        demo.launch(share=args.share)


if __name__ == "__main__":
    import logging
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    main()
