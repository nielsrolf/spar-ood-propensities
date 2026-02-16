"""
Test LocalRouterRunner implementation
"""
import asyncio
import os
from vibes_eval.runner import LocalRouterRunner

async def test_localrouter_runner():
    """Test basic LocalRouterRunner functionality"""
    
    # Check if we have any API keys configured
    has_api_key = any([
        'OPENAI_API_KEY' in os.environ,
        'ANTHROPIC_API_KEY' in os.environ,
        'GEMINI_API_KEY' in os.environ,
        'OPENROUTER_API_KEY' in os.environ
    ])
    
    if not has_api_key:
        print("⚠️  No API keys found. Skipping test.")
        print("Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY")
        return
    
    # Create runner
    runner = LocalRouterRunner(parallel_requests=10)
    print("✅ LocalRouterRunner initialized")
    
    # Create test batch
    questions = [
        "What is 2+2?",
        "What color is the sky?",
        "Name a planet."
    ]
    
    batch = [
        {
            "messages": [{"role": "user", "content": q}],
            "max_tokens": 50,
            "temperature": 0.7
        }
        for q in questions
    ]
    
    # Test with a small, fast model
    model = "gpt-4.1-mini"  # Fast and cheap
    print(f"\n🔄 Running inference with {model}...")
    
    results = await runner.inference(
        model=model,
        questions=questions,
        batch=batch
    )
    
    # Verify results
    assert len(results) == len(questions), "Should return same number of results"
    
    print("\n✅ Results:")
    for result in results:
        print(f"  Q: {result['question']}")
        print(f"  A: {result['answer'][:100]}...")
        print()
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    asyncio.run(test_localrouter_runner())