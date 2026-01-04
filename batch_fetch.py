"""Batch fetch data for all tickers in the assets folder."""

import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Config file to use for fetching
CONFIG_PATH = Path("config/multifetch.yml")

# Template for the config file
CONFIG_TEMPLATE = """data:
  ticker: {ticker}
  start: '2017-01-01'
  interval: 1d
  data_source: alphavantage
"""


@dataclass
class FetchResult:
    """Result of a single fetch operation."""
    ticker: str
    success: bool
    duration: float = 0.0
    rows_fetched: int = 0
    error_message: str = ""
    output: str = ""


@dataclass
class BatchLog:
    """Log of all fetch operations."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    results: list[FetchResult] = field(default_factory=list)
    
    def add(self, result: FetchResult):
        self.results.append(result)
    
    @property
    def successes(self) -> list[FetchResult]:
        return [r for r in self.results if r.success]
    
    @property
    def failures(self) -> list[FetchResult]:
        return [r for r in self.results if not r.success]
    
    @property
    def total_duration(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return sum(r.duration for r in self.results)
    
    def print_summary(self):
        """Print a detailed summary log."""
        self.end_time = datetime.now()
        
        print("\n")
        print("=" * 70)
        print("BATCH FETCH LOG")
        print("=" * 70)
        print(f"Started:  {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Finished: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {self.total_duration:.1f}s")
        print()
        
        # Success summary
        print(f"SUCCESSFUL: {len(self.successes)}/{len(self.results)}")
        print("-" * 70)
        if self.successes:
            for r in self.successes:
                rows_info = f"({r.rows_fetched} rows)" if r.rows_fetched > 0 else ""
                print(f"  [OK] {r.ticker:<8} {r.duration:>6.1f}s  {rows_info}")
        else:
            print("  (none)")
        print()
        
        # Failure summary
        print(f"FAILED: {len(self.failures)}/{len(self.results)}")
        print("-" * 70)
        if self.failures:
            for r in self.failures:
                print(f"  [X] {r.ticker:<8} {r.duration:>6.1f}s")
                if r.error_message:
                    # Indent error message
                    for line in r.error_message.split('\n')[:3]:  # First 3 lines
                        if line.strip():
                            print(f"       -> {line.strip()[:60]}")
        else:
            print("  (none)")
        
        print()
        print("=" * 70)


def get_asset_tickers() -> list[str]:
    """Get all ticker folder names from assets directory."""
    assets_dir = Path("assets")
    if not assets_dir.exists():
        print("Error: assets directory not found")
        sys.exit(1)
    
    tickers = []
    for item in assets_dir.iterdir():
        if item.is_dir() and item.name != "PORTFOLIO":
            tickers.append(item.name)
    
    return sorted(tickers)


def update_config(ticker: str):
    """Update the multifetch.yml config with the given ticker."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(CONFIG_TEMPLATE.format(ticker=ticker))


def fetch_ticker(ticker: str) -> FetchResult:
    """Run the fetch command for a ticker. Returns FetchResult."""
    print(f"\n{'='*60}")
    print(f"Fetching: {ticker}")
    print('='*60)
    
    start = time.time()
    result = FetchResult(ticker=ticker, success=False)
    
    try:
        update_config(ticker)
    except Exception as e:
        result.error_message = f"Config error: {e}"
        result.duration = time.time() - start
        print(f"  ERROR: Failed to write config - {e}")
        return result
    
    cmd = [sys.executable, "main.py", "--config", str(CONFIG_PATH), "fetch", "--force"]
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per ticker
        )
        
        result.output = proc.stdout
        result.duration = time.time() - start
        
        # Print stdout
        if proc.stdout:
            print(proc.stdout)
        
        # Check for errors
        if proc.returncode != 0:
            result.success = False
            result.error_message = proc.stderr or f"Exit code: {proc.returncode}"
            if proc.stderr:
                print(f"  STDERR: {proc.stderr}")
        else:
            result.success = True
            # Try to extract rows fetched from output
            for line in proc.stdout.split('\n'):
                if 'Fetched' in line and 'rows' in line:
                    try:
                        # "Fetched 1234 rows for..."
                        parts = line.split()
                        idx = parts.index('Fetched') + 1
                        result.rows_fetched = int(parts[idx])
                    except (ValueError, IndexError):
                        pass
                    break
        
    except subprocess.TimeoutExpired:
        result.duration = time.time() - start
        result.error_message = "Timeout (exceeded 5 minutes)"
        print(f"  ERROR: Fetch timed out after 5 minutes")
        
    except subprocess.SubprocessError as e:
        result.duration = time.time() - start
        result.error_message = f"Subprocess error: {e}"
        print(f"  ERROR: {e}")
        
    except Exception as e:
        result.duration = time.time() - start
        result.error_message = f"Unexpected error: {e}"
        print(f"  ERROR: Unexpected error - {e}")
    
    return result


def main():
    tickers = get_asset_tickers()
    
    if not tickers:
        print("No tickers found in assets folder")
        sys.exit(1)
    
    print(f"Found {len(tickers)} tickers to fetch:")
    print(f"  {', '.join(tickers)}")
    
    log = BatchLog()
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}]", end="")
        result = fetch_ticker(ticker)
        log.add(result)
        
        # Brief status after each fetch
        if result.success:
            print(f"  [OK] {ticker} complete")
        else:
            print(f"  [X] {ticker} failed - continuing to next...")
    
    # Print detailed log
    log.print_summary()
    
    # Exit with error code if any failures
    if log.failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
