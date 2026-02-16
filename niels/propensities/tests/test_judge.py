"""Test the new LocalRouterJudge0to100"""
import asyncio
from vibes_eval.judge import LocalRouterJudge0to100, OpenAiJudge0to100, free_form_judge_0_100

async def test_localrouter_judge():
    print("Testing LocalRouterJudge0to100...")
    
    # Simple prompt template
    prompt_template = [
        {"role": "user", "content": "Rate the following response on a scale of 0-100 for quality. Response: {response}"}
    ]
    
    # Create judge with gpt-4o-mini (cheaper for testing)
    judge = LocalRouterJudge0to100(
        model="gpt-4o-mini",
        prompt_template=prompt_template,
        n_samples=3  # Use 3 samples for faster testing
    )
    
    # Test with a good response
    test_values = {
        "response": "The capital of France is Paris. It's a beautiful city with rich history and culture."
    }
    
    score = await judge.judge(**test_values)
    print(f"Score for good response: {score}")
    assert score is not None, "Score should not be None"
    assert 0 <= score <= 100, f"Score {score} should be between 0 and 100"
    
    # Test with a bad response
    bad_values = {
        "response": "I don't know, maybe something something blah blah."
    }
    
    score_bad = await judge.judge(**bad_values)
    print(f"Score for bad response: {score_bad}")
    assert score_bad is not None, "Score should not be None"
    assert 0 <= score_bad <= 100, f"Score {score_bad} should be between 0 and 100"
    
    print("✓ LocalRouterJudge0to100 tests passed!")

async def test_factory_function():
    print("\nTesting factory function...")
    
    prompt_template = [
        {"role": "user", "content": "Rate this: {response}"}
    ]
    
    # Test auto selection for OpenAI model (should use logprob)
    judge_openai = free_form_judge_0_100(
        model="gpt-4o-mini",
        prompt_template=prompt_template,
        judge_type="auto"
    )
    assert isinstance(judge_openai, OpenAiJudge0to100), "Should create OpenAiJudge0to100 for OpenAI model"
    print("✓ Auto selection works for OpenAI model")
    
    # Test explicit sampling
    judge_sampling = free_form_judge_0_100(
        model="gpt-4o-mini",
        prompt_template=prompt_template,
        judge_type="sampling",
        n_samples=2
    )
    assert isinstance(judge_sampling, LocalRouterJudge0to100), "Should create LocalRouterJudge0to100 for sampling"
    print("✓ Explicit sampling works")
    
    # Test it can actually judge
    score = await judge_sampling.judge(response="Good answer")
    assert score is not None and 0 <= score <= 100
    print(f"✓ Sampling judge works: score = {score}")

async def main():
    await test_localrouter_judge()
    await test_factory_function()
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    asyncio.run(main())