import pytest

from complaint_routing import config
from complaint_routing.llm import settings
from complaint_routing.llm.classify import UNPARSED, parse_queue
from complaint_routing.llm.evaluate import judge_verdict
from complaint_routing.llm.index import chunk_corpus, chunk_words
from complaint_routing.llm.metrics import first_relevant_rank, hit_rate, mrr

QUEUES = config.queues()


def test_chunks_cover_every_word_with_the_configured_overlap():
    words = [f"w{i}" for i in range(250)]
    chunks = chunk_words(" ".join(words), size=100, overlap=20)
    assert [len(c.split()) for c in chunks] == [100, 100, 90]
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]
    assert chunks[-1].split()[-1] == "w249"


def test_short_document_is_one_chunk_and_bad_overlap_is_rejected():
    assert chunk_words("a b c", size=100, overlap=20) == ["a b c"]
    with pytest.raises(ValueError):
        chunk_words("a b c", size=10, overlap=10)


def test_chunk_ids_are_document_scoped():
    docs = [{"doc_id": "ask-1", "title": "t", "url": "u", "text": " ".join(["x"] * 150)}]
    assert [c["chunk_id"] for c in chunk_corpus(docs, 100, 20)] == ["ask-1#0", "ask-1#1"]


def test_retrieval_metrics_by_hand():
    # q1: relevant doc at rank 1, q2: at rank 3 (two chunks of other docs first), q3: missed.
    ranks = [
        first_relevant_rank(["a", "b"], {"a"}),
        first_relevant_rank(["x", "x", "b", "b"], {"b"}),
        first_relevant_rank(["x", "y"], {"z"}),
    ]
    assert ranks == [1, 3, None]
    assert hit_rate(ranks, 1) == pytest.approx(1 / 3)
    assert hit_rate(ranks, 3) == pytest.approx(2 / 3)
    assert mrr(ranks, 10) == pytest.approx((1 + 1 / 3) / 3)
    assert mrr(ranks, 2) == pytest.approx(1 / 3)


def test_queue_parsing():
    assert parse_queue("debt_collection", QUEUES) == "debt_collection"
    assert parse_queue("Queue: Checking savings.", QUEUES) == "checking_savings"
    assert parse_queue("mortgage, not card", QUEUES) == "mortgage"
    assert parse_queue("I cannot tell", QUEUES) == UNPARSED


def test_judge_verdict():
    assert judge_verdict("YES") == 1
    assert judge_verdict("no.") == 0
    assert judge_verdict("Maybe") is None
    assert judge_verdict("") is None


def test_prompt_hash_changes_when_the_file_changes(tmp_path, monkeypatch):
    cfg = settings.load()
    prompt = tmp_path / "p.txt"
    prompt.write_text("Question: $question")
    monkeypatch.setattr(settings, "ROOT", tmp_path)
    cfg["prompts"]["answer"] = "p.txt"
    before = settings.prompt_hash(cfg, "answer")
    assert settings.render(cfg, "answer", question="q?") == "Question: q?"
    prompt.write_text("Question: $question\nAnswer briefly.")
    assert settings.prompt_hash(cfg, "answer") != before


def test_overrides_are_typed_and_change_the_config_hash():
    base = settings.load()
    cfg = settings.load(overrides=["retrieval.chunk_size=60", "judge.model=null"])
    assert cfg["retrieval"]["chunk_size"] == 60 and cfg["judge"]["model"] is None
    assert settings.config_hash(cfg) != settings.config_hash(base)
    with pytest.raises(KeyError):
        settings.load(overrides=["retrieval.chunk_sise=60"])


def test_hand_grades_count_only_for_the_exact_answer():
    from complaint_routing.llm.evaluate import answer_sha, hand_metrics

    rows = [
        {"qid": "q01", "answer": "No, it does not.", "correct": 1},
        {"qid": "q02", "answer": "Yes.", "correct": 1},
        {"qid": "q03", "answer": "changed answer", "correct": 0},
    ]
    grades = {("q01", answer_sha("No, it does not.")): 1, ("q02", answer_sha("Yes.")): 0,
              ("q03", answer_sha("old answer")): 1}  # fmt: skip
    m = hand_metrics(rows, grades)
    assert m["hand_graded_n"] == 2
    assert m["answer_correct_hand"] == 0.5
    assert m["judge_agreement_hand"] == 0.5 and m["judge_false_yes_hand"] == 0.5
