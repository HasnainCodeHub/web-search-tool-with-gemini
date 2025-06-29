import chainlit as cl,os,requests
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
from agents.tool import function_tool
from agents.run import RunConfig
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent


# Load environment variables
load_dotenv()

# Get Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Disable tracing
set_tracing_disabled(disabled=True)

# Configure external client for Gemini API
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Define the model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Configure run settings
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Define the web search tool
@function_tool("web_search_tool")
def web_search_tool(query: str, num_results: int = 5):
    """
    Perform a Google search using the Serper API and return top result summaries with source URLs.
    """
    print("Tool Message: Web Search Tool is Called!")
    print("=" * 40)

    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return [{
            "title": "Configuration Error",
            "url": "",
            "summary": "Serper API key is not set. Please set the SERPER_API_KEY environment variable."
        }]

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "q": query,
        "num": num_results
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            results = response.json()
            organic_results = results.get('organic', [])
            if not organic_results:
                return [{
                    "title": "No Results",
                    "url": "",
                    "summary": "No results found for the search query."
                }]

            results_summary = []
            for result in organic_results[:num_results]:
                title = result.get('title', 'No Title')
                url = result.get('link', '')
                snippet = result.get('snippet', 'No summary available')
                summary = f"{snippet}\n\nSource: {url}"
                results_summary.append({
                    "title": title,
                    "url": url,
                    "summary": summary
                })
            return results_summary
        else:
            return [{
                "title": "API Error",
                "url": "",
                "summary": f"Error {response.status_code}: {response.text}"
            }]
    except requests.exceptions.Timeout:
        return [{
            "title": "Request Timeout",
            "url": "",
            "summary": "The request timed out. Please try again later."
        }]
    except requests.exceptions.RequestException as e:
        return [{
            "title": "Request Error",
            "url": "",
            "summary": str(e)
        }]

# Create the agent
agent = Agent(
    name="Assistant",
    instructions="You Are a Helpful Assistant.",
    tools=[web_search_tool],
    model=model
)


@cl.on_chat_start
async def start():
    # Initialize conversation history
    cl.user_session.set('history', [])
    # Send welcome message
    await cl.Message(content="Web Search Tool").send()

@cl.on_message
async def handle_message(message: cl.Message):
    # Get conversation history
    history = cl.user_session.get('history')
    
    # Initialize response message with streaming enabled
    msg = cl.Message(content="")
    await msg.send()

    # Append user message to history
    history.append({"role": "user", "content": message.content})

    # Run the agent with the history as input
    result = Runner.run_streamed(
        agent,
        input=history,
        run_config=config
    )

    # Stream response events
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    # Append assistant response to history
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set('history', history)

    # Finalize the response
    await msg.update()