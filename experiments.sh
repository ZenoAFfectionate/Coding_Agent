#!/usr/bin/env bash
# ============================================================================
# experiments.sh — Run all evaluation benchmarks for the Coding Agent
#
# Model: Qwen3.5-30B-A3B (served via vLLM)
# Agent types: react (ReActAgent), funca (FunctionCallAgent)
# Benchmarks: BFCL, SWE-bench Verified, TritonBench, GAIA
#
# Usage:
#   bash experiments.sh 2>&1 | tee experiments_output.txt
#
# All results are saved to results/ as JSON files.
# Console output (including progress and summaries) is captured in output.txt.
# ============================================================================

set -euo pipefail

OUTPUT_FILE="experiments_output.txt"
RESULTS_DIR="results"
TEMPERATURE=0.2
MAX_ITER=32
SWARM_WORKER_STEPS=20
SWARM_ROUNDS=8

mkdir -p "$RESULTS_DIR"

# Redirect all output to both console and file
exec > >(tee -a "$OUTPUT_FILE") 2>&1

echo "============================================================"
echo "  Coding Agent — Experiment Suite"
echo "  Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Output file: $OUTPUT_FILE"
echo "============================================================"
echo ""

# --------------------------------------------------------------------------
# 1. BFCL — Function Calling Accuracy (direct LLM, no agent loop)
#    19 categories total; we test all meaningful ones grouped by type.
# --------------------------------------------------------------------------
echo "================================================================"
echo "  [1/4] BFCL — Function Calling Accuracy"
echo "================================================================"

# Core single-turn categories
BFCL_CORE="simple_python simple_java simple_javascript multiple parallel parallel_multiple"
# Live (real-world) categories
BFCL_LIVE="live_simple live_multiple live_parallel live_parallel_multiple"
# Multi-turn & agentic categories
BFCL_MULTI="multi_turn_base multi_turn_miss_func multi_turn_miss_param multi_turn_long_context"

for CATEGORY in $BFCL_MULTI; do
    echo ""
    echo "--- BFCL: $CATEGORY ---"
    python evaluation.py \
        --benchmark bfcl \
        --category "$CATEGORY" \
        --temperature "$TEMPERATURE" \
        --output "$RESULTS_DIR/bfcl_${CATEGORY}_results.json" \
        --export \
        || echo "[WARN] BFCL $CATEGORY failed, continuing..."
done

# --------------------------------------------------------------------------
# 2. SWE-bench Verified — Real-World Issue Resolution
# --------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  [2/4] SWE-bench Verified — Issue Resolution"
echo "================================================================"

for AGENT_TYPE in react funca; do
    echo ""
    echo "--- SWEV: agent=$AGENT_TYPE ---"
    python evaluation.py \
        --benchmark swev \
        --agent-type "$AGENT_TYPE" \
        --max-iterations "$MAX_ITER" \
        --temperature "$TEMPERATURE" \
        --output "$RESULTS_DIR/swev_${AGENT_TYPE}_results.json" \
        --export \
        || echo "[WARN] SWEV $AGENT_TYPE failed, continuing..."
done

# SWE-bench with Agent Swarm (multi-agent) — uses run_swarm.py directly
echo ""
echo "--- SWEV: Agent Swarm (multi-agent) ---"
echo "[NOTE] Agent Swarm evaluation for SWE-bench requires manual setup."
echo "       Use: python run_swarm.py --task '<issue_description>' --workspace <repo_path>"

# --------------------------------------------------------------------------
# 3. TritonBench — GPU Kernel Generation (G channel, complex instructions)
# --------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  [3/4] TritonBench — GPU Kernel Generation"
echo "================================================================"

for AGENT_TYPE in react funca; do
    echo ""
    echo "--- TritonBench: agent=$AGENT_TYPE, channel=G, mode=complex ---"
    python evaluation.py \
        --benchmark trib \
        --agent-type "$AGENT_TYPE" \
        --channel G \
        --instruction-mode complex \
        --max-iterations "$MAX_ITER" \
        --temperature "$TEMPERATURE" \
        --output "$RESULTS_DIR/trib_G_complex_${AGENT_TYPE}_results.json" \
        || echo "[WARN] TritonBench $AGENT_TYPE failed, continuing..."
done

# --------------------------------------------------------------------------
# 4. GAIA — General AI Assistant Capability
# --------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  [4/4] GAIA — General AI Assistant"
echo "================================================================"

for LEVEL in 1 2 3; do
    echo ""
    echo "--- GAIA: level=$LEVEL ---"
    python evaluation.py \
        --benchmark gaia \
        --level "$LEVEL" \
        --temperature "$TEMPERATURE" \
        --output "$RESULTS_DIR/gaia_level${LEVEL}_results.json" \
        || echo "[WARN] GAIA level $LEVEL failed, continuing..."
done

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  All experiments completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""
echo "Results saved to:"
ls -lh "$RESULTS_DIR"/*.json 2>/dev/null || echo "  (no result files found)"
echo ""
echo "Full output log: $OUTPUT_FILE"
