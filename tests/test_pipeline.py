"""
Tests for the Q&A Pipeline.

Run: pytest tests/ -v
"""

import os

import pytest
from transformers import pipeline as hf_pipeline

import src.pipeline as pipeline_module
from src.knowledge_base import build_knowledge_base
from src.pipeline import ask_question, build_arg_parser, get_llm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture(scope="module")
def vector_store():
    """Build the vector store once for all tests."""
    return build_knowledge_base(DATA_DIR)


@pytest.fixture(scope="module")
def llm():
    """Load the LLM once for all tests."""
    return get_llm()


# ────────────────────────────────
# ask_question return structure
# ────────────────────────────────
class TestAskQuestionStructure:
    def test_returns_dict(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What services do you offer?")
        assert isinstance(result, dict), "ask_question should return a dict"

    def test_has_answer_key(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What services do you offer?")
        assert "answer" in result, "Result dict must have an 'answer' key"

    def test_has_sources_key(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What services do you offer?")
        assert "sources" in result, "Result dict must have a 'sources' key"

    def test_answer_is_string(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What services do you offer?")
        assert isinstance(result["answer"], str), "'answer' should be a string"
        assert len(result["answer"].strip()) > 0, "'answer' should not be empty"

    def test_sources_is_list(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What services do you offer?")
        assert isinstance(result["sources"], list), "'sources' should be a list"
        assert len(result["sources"]) > 0, "'sources' should not be empty"


# ────────────────────────────────
# Retrieval quality
# ────────────────────────────────
class TestRetrieval:
    def test_retrieves_pricing_info(self, vector_store, llm):
        result = ask_question(vector_store, llm, "How much does the Growth package cost?")
        sources_text = " ".join(result["sources"]).lower()
        assert "growth" in sources_text or "$5,500" in sources_text, (
            "Sources should contain pricing-related content"
        )

    def test_retrieves_seo_info(self, vector_store, llm):
        result = ask_question(vector_store, llm, "Do you offer SEO services?")
        sources_text = " ".join(result["sources"]).lower()
        assert "seo" in sources_text or "keyword" in sources_text, (
            "Sources should contain SEO-related content"
        )

    def test_different_questions_get_different_sources(self, vector_store, llm):
        r1 = ask_question(vector_store, llm, "How does onboarding work?")
        r2 = ask_question(vector_store, llm, "What are your PPC management fees?")
        assert r1["sources"] != r2["sources"], (
            "Different questions should retrieve different chunks"
        )


# ────────────────────────────────
# Answer generation
# ────────────────────────────────
class TestAnswerGeneration:
    def test_answer_is_not_just_the_prompt(self, vector_store, llm):
        result = ask_question(vector_store, llm, "Can I cancel my contract?")
        assert "Context:" not in result["answer"], (
            "Answer should be the generated text, not the full prompt"
        )

    def test_answer_responds_to_question(self, vector_store, llm):
        result = ask_question(vector_store, llm, "How much is the Starter package?")
        answer = result["answer"].lower()
        assert "2,500" in answer or "2500" in answer or "starter" in answer, (
            "Answer should address the pricing question"
        )


# ────────────────────────────────
# Error handling additional tests 
# ────────────────────────────────
class TestErrorHandling:
    def test_empty_question_raises(self, vector_store, llm):
        with pytest.raises(ValueError):
            ask_question(vector_store, llm, "")

    def test_whitespace_only_question_raises(self, vector_store, llm):
        with pytest.raises(ValueError):
            ask_question(vector_store, llm, "   ")

    def test_main_handles_missing_data_dir_gracefully(self, monkeypatch, capsys):
        monkeypatch.setattr(pipeline_module.os.path, "isdir", lambda path: False)
        pipeline_module.main()  # should not raise
        captured = capsys.readouterr()
        assert "data directory not found" in captured.out.lower()


# ────────────────────────────────
# Retrieval quality additional tests 
# ────────────────────────────────
class TestRetrievalBonus:
    def test_retrieves_from_unrelated_handbook_doc(self, vector_store, llm):
        """data/company_handbook.txt is unrelated HR content mixed into data/.
        The retriever should still surface it when it's actually the best match."""
        result = ask_question(vector_store, llm, "How many days of PTO do employees get?")
        sources_text = " ".join(result["sources"]).lower()
        assert "pto" in sources_text or "paid time off" in sources_text, (
            "Sources should contain PTO-related content from the handbook doc"
        )

    def test_does_not_confuse_similar_pricing_across_docs(self, vector_store, llm):
        """data/pricing.txt (agency packages) and data/product_faq.txt (AcmeCloud)
        both mention dollar amounts. A Growth package question shouldn't pull
        AcmeCloud pricing instead."""
        result = ask_question(vector_store, llm, "How much does the Growth package cost?")
        sources_text = " ".join(result["sources"]).lower()
        assert "acmecloud" not in sources_text, (
            "Growth package query should not retrieve unrelated AcmeCloud pricing"
        )


# ────────────────────────────────
# CLI argument parsing additional tests
# ────────────────────────────────
class TestArgParser:
    def test_query_flag_parses(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--query", "How much is Growth?"])
        assert args.query == "How much is Growth?"

    def test_query_short_flag_parses(self):
        parser = build_arg_parser()
        args = parser.parse_args(["-q", "Do you offer SEO?"])
        assert args.query == "Do you offer SEO?"

    def test_query_defaults_to_none(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.query is None