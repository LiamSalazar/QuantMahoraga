DATABASE_URL ?= postgresql:///mahoraga_dss
DSS_BACKEND ?= postgres
API_PORT ?= 8002

.PHONY: pipeline adaptive-dry-run adaptive validate api frontend benchmark-queries partition-demo load-test py-compile validate-scalable

pipeline:
	DSS_BACKEND=$(DSS_BACKEND) DATABASE_URL="$(DATABASE_URL)" python -m etl.run_pipeline --mode postgres --profile standard --truncate

adaptive-dry-run:
	DSS_BACKEND=$(DSS_BACKEND) DATABASE_URL="$(DATABASE_URL)" python -m etl.run_adaptive --strategy auto --dry-run

adaptive:
	DSS_BACKEND=$(DSS_BACKEND) DATABASE_URL="$(DATABASE_URL)" python -m etl.run_adaptive --strategy auto

validate:
	DSS_BACKEND=$(DSS_BACKEND) DATABASE_URL="$(DATABASE_URL)" python -m etl.validate_postgres
	DSS_BACKEND=$(DSS_BACKEND) DATABASE_URL="$(DATABASE_URL)" python -m scripts.smoke_postgres

api:
	DSS_BACKEND=$(DSS_BACKEND) DATABASE_URL="$(DATABASE_URL)" uvicorn api.main:app --host 127.0.0.1 --port $(API_PORT)

frontend:
	cd frontend && npm run build

benchmark-queries:
	DATABASE_URL="$(DATABASE_URL)" python -m scripts.benchmark_queries --smoke

partition-demo:
	DATABASE_URL="$(DATABASE_URL)" python -m scripts.partition_pruning_demo

load-test:
	python -m scripts.load_test_api --base-url http://127.0.0.1:$(API_PORT) --concurrency 20 --requests 1000

py-compile:
	python -m py_compile etl/adaptive_planner.py etl/run_adaptive.py etl/incremental.py etl/source_manifest.py etl/partitioned_parquet.py etl/data_contracts.py etl/control_plane.py
	python -m py_compile scripts/generate_scale_fixture.py scripts/benchmark_queries.py scripts/partition_pruning_demo.py scripts/load_test_api.py

validate-scalable: py-compile
	DSS_BACKEND=$(DSS_BACKEND) DATABASE_URL="$(DATABASE_URL)" python -m etl.validate_postgres
	DSS_BACKEND=$(DSS_BACKEND) DATABASE_URL="$(DATABASE_URL)" python -m scripts.smoke_postgres
	DSS_BACKEND=$(DSS_BACKEND) DATABASE_URL="$(DATABASE_URL)" python -m etl.run_adaptive --strategy auto --dry-run
	DATABASE_URL="$(DATABASE_URL)" python -m scripts.benchmark_queries --smoke
	DATABASE_URL="$(DATABASE_URL)" python -m scripts.partition_pruning_demo
