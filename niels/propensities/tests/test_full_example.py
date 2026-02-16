"""
Test the full example from the repository with LocalRouterRunner
"""
import asyncio
import os
from vibes_eval import FreeformQuestion

async def test_full_example():
    """Test FreeformQuestion with LocalRouterRunner"""
    
    # Check if we have any API keys configured
    has_api_key = any([
        'OPENAI_API_KEY' in os.environ,
        'ANTHROPIC_API_KEY' in os.environ,
        'GEMINI_API_KEY' in os.environ,
        'OPENROUTER_API_KEY' in os.environ
    ])
    
    if not has_api_key:
        print("⚠️  No API keys found. Skipping test.")
        return
    
    # Create a freeform question similar to the example
    question = FreeformQuestion(
        id="test_capital",
        paraphrases=[
            "What is the capital of France?",
            "Can you tell me the capital city of France?"
        ],
        samples_per_paraphrase=2,
        temperature=0.7,
        max_tokens=100,
        judge="gpt-4o-2024-08-06",
        judge_prompts={
            "correctness": """Rate how correct and complete this answer is on a scale of 0-100.
            
Question: {question}
Answer: {answer}

Answer with just a number between 0 and 100."""
        },
        results_dir="/tmp/test_full_results"
    )
    
    print("✅ FreeformQuestion created")
    
    # Define models to test
    models = {
        "fast_models": ["gpt-4.1-mini", "gpt-4.1-nano"]
    }
    
    print(f"\n🔄 Running evaluation...")
    
    # Run the evaluation
    results = await question.run(models)
    
    print("\n✅ Results:")
    print(results.df[['model', 'group', 'question', 'answer', 'correctness']].to_string())
    
    print(f"\n📊 Summary statistics:")
    print(results.df.groupby('model')['correctness'].describe())
    
    # Test that we got expected results
    assert len(results.df) == 8  # 2 paraphrases * 2 samples_per_paraphrase = 4 per model, * 2 models = 8
    print("\n✅ Full example test passed!")

if __name__ == "__main__":
    asyncio.run(test_full_example())