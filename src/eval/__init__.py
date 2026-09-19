"""Evaluation module for autoregressive greedy decoding and exact-match benchmarking."""

from src.eval.evaluate import greedy_decode, evaluate_exact_match, run_evaluation

__all__ = ["greedy_decode", "evaluate_exact_match", "run_evaluation"]
