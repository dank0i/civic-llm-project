# civic-llm-project
GitHub Repository for get-civic.com Candidacy Project

# Overview
This is a political AI chatbot that uses reasoning, prompt engineering, and conversational intelligence to respond to general queries on specifically political topics.

The chatbot was written in Python, and uses OpenAIs API to answer queries. It also uses Tavily API Search to evaluate sources and topics, and ensure information quality. 

The code and prompts were written specifically to prioritize balance and neutrality, mitigate bias in responses (especially when the query itself is biased), and admit when there is uncertainty. The chatbot also has its own checks for hallucination prevention.

There is also a debugging / testing suite, with confidence scores and tools like individual step logging.

# Requirements
Python 3 or higher

OpenAI API Key (any tier will do)

Tavily API Key (free tier was used in testing)

# Installation
Clone the repository, and install dependencies
```bash
$ git clone https://github.com/dank0i/civic-llm-project
$ cd civic-llm-project
$ pip install openai
$ pip install gradio
$ pip install aiohttp
```

Then, set up API keys 
```bash
$ export OPENAI_API_KEY = "your-key"
$ export TAVILY_API_KEY = "your-key"
```

Finally, run the program using `python main.py`.
The web interface will open on `http://localhost:7860`.
