from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,set_tracing_disabled
from agents.tool import function_tool
from agents.run import RunConfig
from dotenv import load_dotenv
import os ,asyncio,requests

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

set_tracing_disabled(disabled=True)

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)





@function_tool("web_search_tool")
def web_search_tool(query: str, num_results: int = 5):
    """
    Perform a Google search using the Serper API and return top result summaries with source URLs.
    """
    print("Tool Message: Web Search Tool is Called!")
    print("=" * 40)

    # Load the Serper API key from environment variable
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return [{
            "title": "Configuration Error",
            "url": "",
            "summary": "Serper API key is not set. Please set the SERPER_API_KEY environment variable."
        }]

    # Serper API endpoint
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
        # Make the API request with a timeout
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

agent = Agent(
    name="Assistant",
    instructions="You Are a Helpful Assistant.",
    tools=[web_search_tool], # add tools here
    model=model
)


async def main(input_text: str):
    result = await Runner.run(agent, input=input_text, run_config=config)
    return result.final_output

if __name__ == "__main__":
    while True:
        user_input = input('Enter Your Query: ')
        if user_input.strip().lower() == 'exit':
            break
        final_output = asyncio.run(main(user_input))
        print(final_output)