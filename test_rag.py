import json
import os
from vault_query import query_vault

EVAL_FILE = "eval_data.json"

def run_tests():
    # 1. Load your specific eval format
    if not os.path.exists(EVAL_FILE):
        print(f"❌ Error: {EVAL_FILE} not found.")
        return

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"🧪 Running Retrieval Evaluation on {len(test_cases)} cases...\n")
    
    score = 0
    total = len(test_cases)

    # 2. Test Loop
    for case in test_cases:
        test_id = case.get("id", "Unknown")
        jd_text = case.get("jd", "")
        expected_files = case.get("relevant_resumes", [])
        
        # Skip empty cases
        if not jd_text or not expected_files:
            continue

        print(f"🔎 Testing Case: {test_id}")
        
        # Run the RAG system silently
        try:
            # We don't care about the text answer, only the context (retrieved files)
            _, retrieved_context = query_vault(jd_text)
        except Exception as e:
            print(f"   ⚠️ System Crash: {e}")
            retrieved_context = []

        # 3. Validation Logic
        # Extract filenames from the retrieved chunks
        retrieved_files = []
        for item in retrieved_context:
            # metadata might be hidden deep in the object depending on how build_kb saved it
            meta = item.get("metadata", {})
            fname = meta.get("filename", "unknown.pdf")
            retrieved_files.append(fname)
        
        # Check if AT LEAST ONE of the expected files was found
        hit = False
        for expected in expected_files:
            if expected in retrieved_files:
                hit = True
                break
        
        # 4. Grading
        if hit:
            print("   ✅ PASS (Found correct resume)")
            score += 1
        else:
            print(f"   ❌ FAIL")
            print(f"      Expected: {expected_files}")
            print(f"      Retrieved: {retrieved_files[:3]}...") # Show top 3 found

        print("-" * 30)

    # 5. Final Report
    accuracy = (score / total) * 100
    print(f"\n📊 Retrieval Accuracy: {score}/{total} ({accuracy:.1f}%)")
    
    if accuracy < 100:
        print("💡 Tip: If accuracy is low, try increasing TOP_K or lowering SIM_THRESHOLD.")

if __name__ == "__main__":
    run_tests()