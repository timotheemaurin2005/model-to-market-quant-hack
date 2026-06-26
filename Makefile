.PHONY: test lint run-fx run-dry dashboard

# Run the test suite
test:
	pytest tests/ -v --tb=short

# Run the linter
lint:
	ruff check .

# Run the live FX mean-reversion core
run-fx:
	python trading_engine/live_trader.py

# Run the execution loop in dry-run mode
run-dry:
	python trading_engine/ops/dry_loop.py

# Run the Streamlit macro dashboard
dashboard:
	streamlit run macro_agent/dashboard/app.py
