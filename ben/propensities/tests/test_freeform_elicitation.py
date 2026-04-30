"""Unit tests for FreeformQuestion / FreeformEval elicitation helpers.

These exercise with_system_prompt / with_few_shot / with_user_prefix at the
message-construction level — no LLM calls, no network.
"""

from vibes_eval.freeform import FreeformEval, FreeformQuestion


def _make_question(**overrides) -> FreeformQuestion:
    defaults = dict(
        id="q0",
        paraphrases=["What should I do?"],
        samples_per_paraphrase=1,
        temperature=1.0,
        judge_prompts={"score": "Rate: {question} / {answer}"},
        results_dir="/tmp/_freeform_test_results",
    )
    defaults.update(overrides)
    return FreeformQuestion(**defaults)


def test_with_system_prompt_sets_system_and_clears_context():
    q = _make_question(context=[{"role": "user", "content": "old"}])
    new = q.with_system_prompt("Be helpful.")
    messages = new.as_messages("actual question?")
    assert messages == [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "actual question?"},
    ]


def test_with_few_shot_appends_examples_on_clean_question():
    q = _make_question()
    new = q.with_few_shot(
        [
            {"user": "u1", "assistant": "a1"},
            {"user": "u2", "assistant": "a2"},
        ]
    )
    messages = new.as_messages("Q?")
    assert messages == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "Q?"},
    ]


def test_with_user_prefix_is_its_own_turn_with_ack():
    q = _make_question()
    new = q.with_user_prefix(
        "As someone who cares deeply about X, ",
        ack="Got it.",
    )
    messages = new.as_messages("What do you recommend?")
    assert messages == [
        {"role": "user", "content": "As someone who cares deeply about X, "},
        {"role": "assistant", "content": "Got it."},
        {"role": "user", "content": "What do you recommend?"},
    ]
    # Paraphrases must be unchanged: the prefix is not inlined.
    assert new.paraphrases == q.paraphrases


def test_with_user_prefix_uses_default_ack():
    q = _make_question()
    new = q.with_user_prefix("Preamble.")
    messages = new.as_messages("Q?")
    assert messages[1] == {
        "role": "assistant",
        "content": FreeformQuestion.DEFAULT_USER_PREFIX_ACK,
    }


def test_with_user_prefix_on_clean_question_has_no_system():
    q = _make_question()
    new = q.with_user_prefix("Preamble.")
    messages = new.as_messages("Q?")
    assert all(m["role"] != "system" for m in messages)


def test_freeform_eval_with_user_prefix_applies_to_all_questions():
    questions = [
        _make_question(id="q0", paraphrases=["first?"]),
        _make_question(id="q1", paraphrases=["second?"]),
    ]
    ev = FreeformEval(questions, name="test_eval")
    new_ev = ev.with_user_prefix("prefix")
    assert len(new_ev.questions) == 2
    for q in new_ev.questions:
        msgs = q.as_messages(q.paraphrases[0])
        assert msgs[0] == {"role": "user", "content": "prefix"}
        assert msgs[1]["role"] == "assistant"
    assert new_ev.name == "test_eval_user_prefix"
