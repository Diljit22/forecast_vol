# -----------------------------------------------------------------------------
#  forecast_vol - developer helpers
# -----------------------------------------------------------------------------
#  Quality gates
#    make dev-install   : pip install -e .
#    make lint          : Ruff autofix + re-lint
#    make typecheck     : mypy --strict
#    make test          : pytest -q
#
#  Market-hours calendar
#    make build-sessions  : create NYSE active-session JSON
#
#  Preprocessing
#    make restrict       : trim raw CSV to NYSE grid
#    make resample       : gap-fill restricted bars and recompute VWAP
#    make build-dataset  : feature engineer + one-hot encode
#    make preprocess     : minute-level cleaning and feature engineering
#
#  Synergy layer
#    make hmm             : per-ticker Hidden Markov Models
#    make gnn             : hyperparameter search for correlation GNN
#    make gnn-snapshots   : rolling GNN embeddings
#    make synergy         : regime labelling + GNN correlation model pipeline
#
#  Stochastic layer
#    make stochastic      : wavelet fractal, EGARCH, RFSV feature fits
#
#  Attention layer
#    make attention-merge : build dataset for Transformer
#    make attention-train : train Transformer volatility predictor
#	 make attention		  : attention model pipeline
#
#  Full pipeline
#    make pipeline        : preprocess -> synergy -> stochastic -> attention
#
#  Pass extra CLI flags to any stage with:
#    make <target> ARGS="--debug --jobs 4"
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Global vars (overridable at CLI)
# --------------------------------------------------------------------------- #
PY      ?= python
ARGS    ?=

# --------------------------------------------------------------------------- #
# Phony targets
# --------------------------------------------------------------------------- #
.PHONY: help dev-install lint typecheck test \
		build-sessions \
        restrict resample build-dataset preprocess \
        hmm gnn gnn-snapshots synergy \
		stochastic \
		attention-merge attention-train attention \
		pipeline

# --------------------------------------------------------------------------- #
# Help
# --------------------------------------------------------------------------- #
help:
	@echo "Common targets:"
	@echo "  dev-install     : pip install -e ."
	@echo "  lint            : Ruff autofix + re-lint"
	@echo "  typecheck       : mypy --strict"
	@echo "  test            : pytest -q"
	@echo ""
	@echo "  build-sessions  : create NYSE active-session JSON"
	@echo ""
	@echo "  restrict        : trim raw CSV to NYSE grid"
	@echo "  resample        : gap-fill restricted bars and recompute VWAP"
	@echo "  build-dataset   : feature engineer + one-hot encode"
	@echo "  preprocess      : minute-level cleaning and feature engineering"
	@echo ""
	@echo "  hmm             : per-ticker Hidden Markov Models"
	@echo "  gnn             : hyperparameter search for correlation GNN"
	@echo "  gnn-snapshots   : rolling GNN embeddings"
	@echo "  synergy         : regime labelling + GNN correlation model pipeline"
	@echo ""
	@echo "  stochastic      : wavelet fractal, EGARCH, RFSV feature fits"
	@echo ""
	@echo "  attention-merge : build dataset for Transformer"
	@echo "  attention-train : train Transformer volatility predictor"
	@echo "  attention       : attention model pipeline"
	@echo ""
	@echo "  pipeline        : preprocess -> synergy -> stochastic -> attention"
	@echo ""
	@echo "Pass extra CLI flags to any stage with:"
	@echo "  make <target> ARGS=\"--debug --jobs 4\""

# --------------------------------------------------------------------------- #
# Quality gates
# --------------------------------------------------------------------------- #
dev-install: ## Install package in editable mode
	$(PY) -m pip install -e .

lint:        ## Ruff autofix then re-lint
	$(PY) -m ruff check --fix .
	$(PY) -m ruff check .

typecheck:  ## Strict mypy on src package
	$(PY) -m mypy --strict src/forecast_vol

test:       ## Run pytest suite
	$(PY) -m pytest -q

# --------------------------------------------------------------------------- #
# Market-hours calendar
# --------------------------------------------------------------------------- #
build-sessions: ## forecast_vol.market.build_sessions
	$(PY) -m forecast_vol.market.build_sessions $(ARGS)

# --------------------------------------------------------------------------- #
# Preprocess layer
# --------------------------------------------------------------------------- #
restrict: ## forecast_vol.preprocess.restrict
	$(PY) -m forecast_vol.preprocess.restrict $(ARGS)

resample: ## forecast_vol.preprocess.resample
	$(PY) -m forecast_vol.preprocess.resample $(ARGS)

build-dataset: ## forecast_vol.preprocess.build_dataset
	$(PY) -m forecast_vol.preprocess.build_dataset $(ARGS)

preprocess: restrict resample build-dataset ## minute-level cleaning and feature engineering

# --------------------------------------------------------------------------- #
# Synergy layer (HMM + GNN)
# --------------------------------------------------------------------------- #
hmm: ## forecast_vol.synergy.hmm_fit
	$(PY) -m forecast_vol.synergy.hmm_fit $(ARGS)

gnn: ## forecast_vol.synergy.optuna_search
	$(PY) -m forecast_vol.synergy.optuna_search $(ARGS)

gnn-snapshots: ## daily rolling embeddings
	$(PY) -m forecast_vol.synergy.gnn_snapshots $(ARGS)

synergy: hmm gnn gnn-snapshots ## regime labelling + GNN correlation model pipeline

# --------------------------------------------------------------------------- #
# Stochastic layer (EGARCH(1,1) + fractal dimension + RFSV)
# --------------------------------------------------------------------------- #
stochastic: ## forecast_vol.stochastic.fit_all
	$(PY) -m forecast_vol.stochastic.fit_all $(ARGS)

# --------------------------------------------------------------------------- #
# Attention (Transformer) layer
# --------------------------------------------------------------------------- #
attention-merge: ## forecast_vol.attention.data
	$(PY) -m forecast_vol.attention.data $(ARGS)

attention-train: ## forecast_vol.attention.train
	$(PY) -m forecast_vol.attention.train $(ARGS)

attention: attention-merge attention-train ## Transformer training pipeline

# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
pipeline: build-sessions preprocess synergy stochastic attention ## End-to-end run

# Default goal
.DEFAULT_GOAL := pipeline
