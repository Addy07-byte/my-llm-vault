import requests

BASE_URL = "http://127.0.0.1:8000"
conversation_history = [] # 1. Initialize the memory list

while True:
    try:
        user_query = input("\nYou: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            break

        # 2. Build the payload for the API
        payload = {
            "question": user_query,
            # PASS THE CURRENT HISTORY TO THE SERVER
            "history": conversation_history 
        }

        # 3. Make the request
        # Note: We send the history, but NOT the current user_query, 
        # as that is handled by the "question" field.
        response = requests.post(f"{BASE_URL}/query", json=payload)
        
        if response.status_code == 200:
            llm_answer = response.json().get("answer")
            print(f"Assistant: {llm_answer}")

            # 4. UPDATE THE HISTORY (CRUCIAL STEP)
            # Add the user's turn
            conversation_history.append({"role": "user", "content": user_query})
            # Add the assistant's turn
            conversation_history.append({"role": "assistant", "content": llm_answer})

        else:
            print(f"\n❌ Error: API returned status code {response.status_code}")
            print(response.json())

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to FastAPI server. Is it running?")
        break
    except KeyboardInterrupt:
        break