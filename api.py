
# Imports
from fastapi import FastAPI, HTTPException
import uuid
from schemas import ChatRequest, ChatResponse
from chatbot import generate_answer_v2


# CREATE FASTAPI APP

app= FastAPI(
    title="Scripture Chatbot API",
    description="Chat about Hindu scriptures",
    version="1.0.0"
)

print("FastAPI app created")

"""***APP IS NOW OUR API NEXT WE'LL ASS "ROUTES"***"""

# CREATING FIRST ENDPOINT -> ROOT

@app.get("/")
async def root():
  """
   The home page of API
   When someone visits: http://localhost:8000/
  """
  return {
      "message": "Welcome to Scripture Chatbot API!",
      "endpoints": {
          "chat": "/chat",
          "health": "/health",
          "docs": "/docs"
      }
  }

print("✅ Root endpoint created")

# CREATING HEALTH CHECK ENDPOINT

@app.get("/health")
async def health_check():
  """
  Check if API is running
  """
  return {
      "status": "Healthy",
      "message":" API is running"

  }

  print("✅ Health check endpoint created")

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint
    """
    try:
        # 1. Get or create session id
        import uuid
        session_id = request.session_id or str(uuid.uuid4())

        # 2. Validate message
        if not request.message or not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # 3. Call the chatbot function
        result = generate_answer_v2(
            query=request.message.strip(),
            session_id=session_id
        )

        # 4. Handle different response types
        if result["type"] == "decline":
            # Out-of-domain: return just the decline message
            return ChatResponse(
                response=result["response"],  # ← This is a string
                type="decline",
                session_id=session_id,
                sources=[],
                confidence=0.0
            )
        else:
            # In-domain: return normal response
            return ChatResponse(
                response=result["response"],
                type=result["type"],
                session_id=session_id,
                sources=result.get("sources", []),
                confidence=result.get("confidence", 0.8)
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

print("✅ Chat endpoint updated")

