"""
Test FreeformQuestion with LocalRouterRunner
"""
import asyncio
import os
from vibes_eval import FreeformQuestion, dispatcher

async def test_freeform_with_localrouter():
    """Test that FreeformQuestion works with LocalRouterRunner"""
    
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
    
    # Create a simple freeform question
    question = FreeformQuestion(
        id="test_question",
        paraphrases=["What is the capital of France?"],
        samples_per_paraphrase=1,
        temperature=0.7,
        max_tokens=50,
        judge_prompts={
            "correctness": "Rate how correct this answer is on a scale of 0-100. Answer with just a number."
        },
        results_dir="/tmp/test_results"
    )
    
    print("✅ FreeformQuestion created")
    
    # Test with a fast model
    model = "gpt-4.1-mini"
    print(f"\n🔄 Running question with {model}...")
    
    # Run the question
    df = await question.run_model(model)
    
    print("\n✅ Results:")
    print(df[['question', 'answer', 'correctness']].to_string())
    
    # Verify results
    assert len(df) == 1, "Should have 1 result"
    assert 'answer' in df.columns, "Should have answer column"
    assert 'correctness' in df.columns, "Should have correctness score"
    
    print("\n✅ Integration test passed!")

if __name__ == "__main__":
    asyncio.run(test_freeform_with_localrouter())