.PHONY: test ablation

TEST_ARGS?=

ablation:
	python backtest/ablation.py --symbols DEMO --start 2023-01-01 --end 2023-12-31 --enable-ta true --mock-ta true

test:
	pytest $(TEST_ARGS)
