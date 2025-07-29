#!/usr/bin/env python3
"""
Run script để test NASCA model sau khi train
"""

import sys
import time
import json
from pathlib import Path

# Import model từ model.py
from model import UltimateNASCAGenModel

def load_training_data():
    """Load dữ liệu training từ data.txt với error handling"""
    print("🔍 DEBUGGING: Starting load_training_data()...")
    training_conversations = []
    error_lines = 0

    try:
        # Try multiple encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open('data.txt', 'r', encoding=encoding) as f:
                    lines = f.readlines()
                print(f"✅ Successfully opened data.txt with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            print("❌ Could not decode data.txt with any encoding")
            return []

        for line_num, line in enumerate(lines, 1):
            try:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                    
                if '|' in line:
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        input_text = parts[0].strip()
                        output_text = parts[1].strip()
                        
                        # Validate non-empty
                        if input_text and output_text:
                            training_conversations.append({
                                'input': input_text,
                                'output': output_text
                            })
                        else:
                            error_lines += 1
                    else:
                        error_lines += 1
                else:
                    error_lines += 1
                    
            except Exception as e:
                print(f"⚠️ Error processing line {line_num}: {e}")
                error_lines += 1
                continue

        print(f"✅ Loaded {len(training_conversations)} training conversations")
        if error_lines > 0:
            print(f"⚠️ Skipped {error_lines} problematic lines")
        return training_conversations

    except FileNotFoundError:
        print("❌ File data.txt not found!")
        return []
    except Exception as e:
        print(f"❌ Error loading data.txt: {e}")
        return []

def build_vocab_from_data(conversations):
    """Xây dựng vocabulary từ data"""
    vocab_set = {'<EOS>', '<UNK>'}
    for conv in conversations:
        vocab_set.update(conv['input'].lower().split())
        vocab_set.update(conv['output'].lower().split())
    return sorted(list(vocab_set))

def train_model():
    """Train model và return model instance"""
    print("🚀 Starting NASCA Model Training...")
    print("=" * 50)
    print("🔍 DEBUGGING: train_model() function called")
    print("🔍 Current working directory:", Path.cwd())

    # Load data
    training_conversations = load_training_data()
    if not training_conversations:
        return None

    # Build vocab
    vocab = build_vocab_from_data(training_conversations)
    intents = ['greet', 'question', 'request', 'farewell']

    print(f"📚 Vocabulary size: {len(vocab)}")
    print(f"🎯 Intents: {len(intents)}")

    # Initialize model
    config = {
        'num_voters': 3,
        'embedding_dim': 12,
        'max_tokens': 20
    }

    print("🔧 Initializing model...")
    model = UltimateNASCAGenModel(vocab, intents, config)

    # Train model
    print("🎓 Training model...")
    start_time = time.time()
    model.train_from_conversations(training_conversations, epochs=3)
    train_time = time.time() - start_time

    print(f"✅ Training completed in {train_time:.2f}s")

    return model, training_conversations

def test_model_responses(model, test_cases):
    """Test model với các test cases"""
    print("\n🧪 Testing Model Responses")
    print("=" * 50)

    results = []

    for i, test_case in enumerate(test_cases, 1):
        input_text = test_case['input']
        expected = test_case['output']

        print(f"\n{i}. Testing: '{input_text}'")
        print(f"   Expected: '{expected}'")

        # Test với different creativity levels
        creativity_levels = [0.3, 0.7, 0.9]

        for creativity in creativity_levels:
            start_time = time.time()
            result = model.generate_response(
                input_text,
                max_tokens=15,
                creativity_level=creativity
            )
            response_time = time.time() - start_time

            creativity_name = ['Low', 'Medium', 'High'][creativity_levels.index(creativity)]
            print(f"   🎨 {creativity_name}: '{result['response']}'")
            print(f"      ⚡ Time: {response_time:.3f}s | Confidence: {result['confidence']:.2f}")

            results.append({
                'input': input_text,
                'expected': expected,
                'response': result['response'],
                'creativity': creativity,
                'time': response_time,
                'confidence': result['confidence'],
                'method': result.get('method', 'unknown')
            })

    # Test same input multiple times để show improved variety
    print(f"🎲 Anti-Repetition Test - Multiple responses cho same input:")
    print("-" * 60)

    test_inputs = ["bạn tên gì", "xin chào", "bạn là ai"]

    for test_input in test_inputs:
        print(f"Input: '{test_input}' (5 lần để test anti-repetition)")
        responses = []

        for i in range(5):
            result = model.generate_response(test_input, creativity_level=0.7)
            response = result['response']
            responses.append(response)
            method = result.get('method', 'unknown')
            attempt = result.get('attempt', 1)
            print(f"   {i+1}. '{response}' [{method}, attempt: {attempt}]")

        # Check uniqueness
        unique_responses = len(set(responses))
        print(f"   📊 Unique responses: {unique_responses}/5 ({unique_responses/5*100:.0f}% variety)")
        print()

    print()

    return results

def interactive_test(model):
    """Interactive testing mode"""
    print("\n💬 Interactive Test Mode")
    print("=" * 30)
    print("Nhập câu để test (hoặc 'quit' để thoát)")
    print("Commands:")
    print("  - 'stats': Xem performance stats")
    print("  - 'creativity X': Set creativity level (0.1-1.0)")
    print("  - 'dream': Force dream cycle")
    print("  - 'dream_stats': Xem dream learning stats")
    print("  - 'dream_knowledge': Xem knowledge từ dreams")
    print("  - 'dream_creativity X': Set dream creativity (0.1-2.0)")
    print("  - 'quit': Thoát")

    creativity_level = 0.7

    while True:
        try:
            user_input = input(f"\n[Creativity:{creativity_level}] 🎯 Input: ").strip()

            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'stats':
                stats = model.get_performance_report()
                print("\n📊 Performance Stats:")
                for key, value in stats.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for sub_key, sub_value in value.items():
                            print(f"    {sub_key}: {sub_value}")
                    else:
                        print(f"  {key}: {value}")
                continue
            elif user_input.lower().startswith('creativity'):
                try:
                    parts = user_input.split()
                    if len(parts) == 2:
                        new_creativity = float(parts[1])
                        if 0.1 <= new_creativity <= 1.0:
                            creativity_level = new_creativity
                            print(f"✅ Creativity level set to {creativity_level}")
                        else:
                            print("❌ Creativity level must be between 0.1 and 1.0")
                    else:
                        print("❌ Usage: creativity 0.7")
                except ValueError:
                    print("❌ Invalid creativity value")
                continue
            elif user_input.lower() == 'dream':
                print("🌙 Triggering dream cycle...")
                dream_stats = model.force_dream_cycle()
                print(f"✨ Dream completed! Stats: {dream_stats}")
                continue
            elif user_input.lower() == 'dream_stats':
                dream_stats = model.lucid_dreamer.get_dream_stats()
                print("\n🌙 Dream Learning Stats:")
                for key, value in dream_stats.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for sub_key, sub_value in value.items():
                            print(f"    {sub_key}: {sub_value:.3f}")
                    else:
                        print(f"  {key}: {value}")
                continue
            elif user_input.lower() == 'dream_knowledge':
                knowledge = model.get_dream_knowledge()
                print(f"\n💭 Dream Knowledge ({len(knowledge)} items):")
                for i, item in enumerate(knowledge[-5:], 1):  # Show last 5
                    print(f"  {i}. {item['type']}: '{item['sequence'][:50]}...' (confidence: {item.get('self_assessed_quality', 0):.2f})")
                continue
            elif user_input.lower().startswith('dream_creativity'):
                try:
                    parts = user_input.split()
                    if len(parts) == 2:
                        new_dream_creativity = float(parts[1])
                        if 0.1 <= new_dream_creativity <= 2.0:
                            model.set_dream_creativity(new_dream_creativity)
                            print(f"✅ Dream creativity set to {new_dream_creativity}")
                        else:
                            print("❌ Dream creativity must be between 0.1 and 2.0")
                    else:
                        print("❌ Usage: dream_creativity 1.5")
                except ValueError:
                    print("❌ Invalid dream creativity value")
                continue
            elif not user_input:
                continue

            # Generate response
            start_time = time.time()
            result = model.generate_response(
                user_input,
                max_tokens=20,
                creativity_level=creativity_level
            )
            response_time = time.time() - start_time

            print(f"🤖 Response: '{result['response']}'")
            print(f"📊 Stats: Time={response_time:.3f}s | Confidence={result['confidence']:.2f} | Method={result.get('method', 'unknown')}")

            # Creative variations demo
            if result.get('method') == 'creative_pattern' or result.get('method') == 'creative_synthesis':
                print(f"✨ Creative mode active!")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def benchmark_performance(model, num_requests=50):
    """Benchmark model performance"""
    print(f"\n⚡ Performance Benchmark ({num_requests} requests)")
    print("=" * 50)

    test_inputs = [
        "xin chào", "bạn tên gì", "giúp tôi", "cảm ơn", "tạm biệt",
        "hello", "bạn là ai", "học python", "machine learning", "deep learning"
    ]

    times = []
    confidences = []

    print("Running benchmark...")
    for i in range(num_requests):
        if i % 10 == 0:
            print(f"  Progress: {i}/{num_requests}")

        test_input = test_inputs[i % len(test_inputs)]

        start_time = time.time()
        result = model.generate_response(test_input, max_tokens=10)
        elapsed = time.time() - start_time

        times.append(elapsed)
        confidences.append(result['confidence'])

    # Calculate stats
    import numpy as np
    avg_time = np.mean(times)
    p95_time = np.percentile(times, 95)
    throughput = num_requests / sum(times)
    avg_confidence = np.mean(confidences)

    print(f"\n📊 Benchmark Results:")
    print(f"  ⚡ Average time: {avg_time:.4f}s")
    print(f"  📈 P95 time: {p95_time:.4f}s")
    print(f"  🚀 Throughput: {throughput:.1f} req/sec")
    print(f"  🎯 Average confidence: {avg_confidence:.3f}")
    print(f"  💾 Memory efficient: {'✅' if avg_time < 0.1 else '⚠️'}")

def main():
    """Main function"""
    print("🎯 NASCA Model Test Runner")
    print("=" * 40)

    # Train model
    model_data = train_model()
    if not model_data:
        print("❌ Failed to train model")
        return

    model, training_conversations = model_data

    # Test with sample data
    test_cases = training_conversations[:5]  # First 5 for quick test
    test_results = test_model_responses(model, test_cases)

    # Performance benchmark
    benchmark_performance(model, num_requests=30)

    # Interactive mode
    print(f"\n🎮 Ready for interactive testing!")
    interactive_test(model)

    # Cleanup
    print(f"\n🧹 Cleaning up...")
    model.graceful_shutdown()
    print(f"✅ Test completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()