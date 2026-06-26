import asyncio
import os
import sys
import json
import yfinance as yf
from dotenv import load_dotenv

# Load environment variables (e.g., GEMINI_API_KEY)
load_dotenv()

# Ensure the correct python path or imports work if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.antigravity import Agent, LocalAgentConfig, types

# --- Custom Tools ---

def fetch_market_data(symbols: list[str]) -> str:
    """Gets the latest market data for a list of Yahoo Finance ticker symbols.
    
    Args:
        symbols: A list of ticker symbols (e.g., ["GC=F", "BTC-USD", "EURUSD=X"]).
    """
    try:
        tickers = yf.download(symbols, period="1d", interval="1d", progress=False)
        if tickers.empty:
            return f"No data found for symbols: {symbols}"
        
        # Format the output into a readable string
        output = []
        for sym in symbols:
            if sym in tickers['Close']:
                close_price = tickers['Close'][sym].iloc[-1]
                output.append(f"{sym}: {close_price:.4f}")
        return "\n".join(output) if output else "Data not available."
    except Exception as e:
        return f"Error fetching market data: {str(e)}"

def get_portfolio_state() -> str:
    """Reads the current live portfolio state (state.json) of the trading engine."""
    state_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "advisory", "state.json")
    # Fallback to the root if not in advisory
    if not os.path.exists(state_file):
        state_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "state.json")
        
    if not os.path.exists(state_file):
        return "state.json not found. The portfolio is currently empty or the engine is not running."
        
    try:
        with open(state_file, "r") as f:
            data = json.load(f)
            return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error reading state.json: {str(e)}"

# --- Orchestrator Configuration ---

async def main():
    skills_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
    
    system_instructions = (
        "You are the Orchestrator for the Model to Market quantitative trading desk. "
        "Your job is to evaluate trading ideas, market context, and risk by coordinating a team of 5 specialized subagents. "
        "Whenever the user asks a question or proposes a trade, you MUST use the `subagents` capability to spawn "
        "and delegate tasks to the following 5 subagents in this order:\n\n"
        "1. Live Data Summariser: Instruct this subagent to use the `fetch_market_data` tool to pull live prices relevant to the user's query.\n"
        "2. Thesis Ingestor: Instruct this subagent to take the user's idea and build the strongest possible bullish/bearish case for it.\n"
        "3. Contradictor: Instruct this subagent to actively attack the Ingestor's thesis, finding flaws, macro headwinds, and structural risks.\n"
        "4. Quant: Instruct this subagent to use the `get_portfolio_state` tool to read the current book, calculate required position sizes, and assess margin impact.\n"
        "5. Judge: Instruct this subagent to evaluate the arguments from the Ingestor, Contradictor, and Quant, and deliver a final Go/No-Go verdict.\n\n"
        "Do not answer the user directly until you have gathered the responses from all 5 subagents. "
        "Your final response to the user should clearly present the findings of each subagent and conclude with the Judge's verdict."
    )

    config = LocalAgentConfig(
        skills_paths=[skills_directory],
        tools=[fetch_market_data, get_portfolio_state],
        capabilities=types.CapabilitiesConfig(
            enable_subagents=True,
        ),
        system_instructions=system_instructions
    )

    print(f"Loading skills from: {skills_directory}")
    print("Booting up the Multi-Agent Orchestrator...\n")
    print("WARNING: Complex queries will spawn 5 subagents. This may take 30-60 seconds to process.\n")
    
    # Instantiate the agent and run the interactive loop in the terminal
    async with Agent(config) as agent:
        await agent.run_interactive_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting Multi-Agent Orchestrator.")
