"""
Debug judge behavior
"""
import asyncio
import os
from vibes_eval.judge import free_form_judge_0_100

async def test_judge():
    """Test the judge directly"""
    
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
    
    # Create a judge
    judge = free_form_judge_0_100(
        model="gpt-4o-2024-08-06",
        prompt_template="Rate how correct this answer is on a scale of 0-100. Answer with just a number.\n\nQuestion: {question}\nAnswer: {answer}"
    )
    
    print(f"✅ Created judge: {type(judge).__name__}")
    
    # Test the judge
    test_data = {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris."
    }
    
    print(f"\n🔄 Judging...")
    score = await judge.judge(**test_data)
    
    print(f"\n✅ Score: {score}")
    print(f"   Type: {type(score)}")

if __name__ == "__main__":
    asyncio.run(test_judge())