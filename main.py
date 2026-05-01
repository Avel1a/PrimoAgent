import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple
import os

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import workflows
from src.workflows.workflow import run_analysis
from src.tools.daily_csv_tool import save_workflow_to_symbol_csv


def get_trading_dates(start_date: str, end_date: str) -> List[str]:
    """Generate list of trading dates (excluding weekends)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    dates = []
    current = start
    while current <= end:
        # Skip weekends (Monday=0, Sunday=6)
        if current.weekday() < 5:  # Monday to Friday
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    return dates


def print_workflow_summary(result: dict, date: str) -> None:
    """Print summary of workflow results for a specific date."""
    if not result.get('success'):
        print(f"{date} - Analysis failed: {result.get('error', 'Unknown error')}")
        return
        
    print(f"{date} - Analysis completed successfully")


def _prompt(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def prompt_symbol_and_dates() -> Tuple[List[str], str, str]:
    """Prompt the user (in English) for a stock symbol and date range.

    Returns:
        (symbols_list, start_date, end_date)
    """
    env_sym = os.getenv("PRIMO_SYMBOL")
    env_start = os.getenv("PRIMO_START_DATE")
    env_end = os.getenv("PRIMO_END_DATE")
    if env_sym and env_start and env_end:
        try:
            datetime.strptime(env_start, "%Y-%m-%d")
            datetime.strptime(env_end, "%Y-%m-%d")
            return [env_sym.strip().upper()], env_start.strip(), env_end.strip()
        except ValueError:
            pass
    # Symbol
    while True:
        sym = _prompt("Enter a stock symbol (e.g., AAPL): ").strip().upper()
        if sym:
            break
        print("Symbol cannot be empty. Example: AAPL")

    # Date parsing helper
    def parse_date(label: str) -> str:
        while True:
            value = _prompt(
                f"Enter {label} date in YYYY-MM-DD format (e.g., 2025-05-28): "
            ).strip()
            try:
                datetime.strptime(value, "%Y-%m-%d")
                return value
            except ValueError:
                print("Invalid date format. Expected YYYY-MM-DD, e.g., 2025-05-28")

    start_date = parse_date("start")
    end_date = parse_date("end")

    # Validate ordering
    while datetime.strptime(start_date, "%Y-%m-%d") > datetime.strptime(end_date, "%Y-%m-%d"):
        print("Start date is after end date. Please re-enter.")
        start_date = parse_date("start")
        end_date = parse_date("end")

    return [sym], start_date, end_date


async def main():
    """Main execution with interactive symbol and date input."""
    try:
        print("PrimoAgent - AI Financial Analysis Platform")
        print("=" * 60)
        
        # Interactive input
        symbols, start_date, end_date = prompt_symbol_and_dates()
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Analysis Period: {start_date} to {end_date}")
        
        # Get trading dates
        trading_dates = get_trading_dates(start_date, end_date)
        print(f"Trading Days: {len(trading_dates)}")
        print(f"Starting workflow execution...")

        if not trading_dates:
            print("No trading days in the selected date range.")
            return
        
        successful_runs = 0
        failed_runs = 0
        company_info_cache = None  # Cache across days (saves Tiingo calls)

        # Process each trading day
        for i, date in enumerate(trading_dates, 1):
            print(f"\n{'='*60}")
            print(f"Day {i}/{len(trading_dates)}: {date}")
            print(f"{'='*60}")

            # Format session ID
            date_formatted = date.replace("-", "_")
            session_id = f"daily_analysis_{date_formatted}"

            try:
                # Execute workflow for this date (pass cached company info to skip Tiingo call)
                result = await run_analysis(symbols, session_id, date,
                                            cached_company_info=company_info_cache)

                # Update company info cache for next day
                if not company_info_cache and result.get('_cached_company_info'):
                    company_info_cache = result['_cached_company_info']
                    print("[CACHE] Company info cached for subsequent days")
                
                # Print results summary
                print_workflow_summary(result, date)
                
                if result.get('success'):
                    successful_runs += 1
                    # Save per-symbol CSV in ./data
                    if save_workflow_to_symbol_csv(result, date, data_dir="./output/csv"):
                        print(f"Per-symbol CSV saved in ./output/csv for {date}")
                else:
                    failed_runs += 1

                # Rate limit: sleep 6s between days to stay within Tiingo free tier (50 req/hr)
                if i < len(trading_dates):
                    time.sleep(6)
                    
            except Exception as e:
                print(f"{date} - Execution error: {e}")
                failed_runs += 1
        
        # Final Summary
        print(f"\nANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Successful Runs: {successful_runs}")
        print(f"Failed Runs: {failed_runs}")
        success_rate = (successful_runs / len(trading_dates) * 100) if trading_dates else 0
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"Main execution error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 
