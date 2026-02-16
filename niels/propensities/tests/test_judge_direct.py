"""Test the new LocalRouterJudge0to100 directly"""
import asyncio
import sys
sys.path.insert(0, '.')

# Import only the judge module
from vibes_eval import judge

async def test_localrouter_judge():
    print("Testing LocalRouterJudge0to100...")
    
    # Simple prompt template
    prompt_template = [
        {"role": "user", "content": "Rate the following response on a scale of 0-100 for quality. Respond with ONLY a number between 0 and 100, nothing else. Response: {response}"}
    ]
    
    # Create judge with gpt-4o-mini (cheaper for testing)
    judge_obj = judge.LocalRouterJudge0to100(
        model="gpt-4o-mini",
        prompt_template=prompt_template,
        n_samples=3  # Use 3 samples for faster testing
    )
    
    # Test with a good response
    test_values = {
        "response": "The capital of France is Paris. It's a beautiful city with rich history and culture."
    }
    
    score = await judge_obj.judge(**test_values)
    print(f"Score for good response: {score}")
    assert score is not None, "Score should not be None"
    assert 0 <= score <= 100, f"Score {score} should be between 0 and 100"
    
    # Test with a bad response
    bad_values = {
        "response": "I don't know, maybe something something blah blah."
    }
    
    score_bad = await judge_obj.judge(**bad_values)
    print(f"Score for bad response: {score_bad}")
    assert score_bad is not None, "Score should not be None"
    assert 0 <= score_bad <= 100, f"Score {score_bad} should be between 0 and 100"
    
    print("✓ LocalRouterJudge0to100 tests passed!")

async def test_factory_function():
    print("\nTesting factory function...")
    
    prompt_template = [
        {"role": "user", "content": "Rate this response for quality from 0-100. Respond ONLY with a number: {response}"}
    ]
    
    # Test auto selection for OpenAI model (should use logprob)
    judge_openai = judge.free_form_judge_0_100(
        model="gpt-4o-mini",
        prompt_template=prompt_template,
        judge_type="auto"
    )
    assert isinstance(judge_openai, judge.OpenAiJudge0to100), "Should create OpenAiJudge0to100 for OpenAI model"
    print("✓ Auto selection works for OpenAI model")
    
    # Test explicit sampling
    judge_sampling = judge.free_form_judge_0_100(
        model="gpt-4o-mini",
        prompt_template=prompt_template,
        judge_type="sampling",
        n_samples=2
    )
    assert isinstance(judge_sampling, judge.LocalRouterJudge0to100), "Should create LocalRouterJudge0to100 for sampling"
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