import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime
from agent import KrishnarajAgent

# FastAPI app
app = FastAPI(title="Krishnaraj's Chatbot", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str


# Initialize the agent
agent = KrishnarajAgent()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the Krishnaraj Resume Bot

    Send a message and get a response about Krishnaraj's professional background,
    skills, experience, and projects.
    """
    try:
        # Process the chat message through the agent
        response = agent.chat(request.message, request.session_id)

        return ChatResponse(
            response=response,
            session_id=request.session_id,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def get_active_sessions():
    """Get information about active chat sessions"""
    try:
        return agent.get_active_sessions()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving sessions: {str(e)}")


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a specific session"""
    try:
        success = agent.clear_session(session_id)
        if success:
            return {"message": f"Session {session_id} cleared successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error clearing session: {str(e)}")



if __name__ == "__main__":
    # Set environment variables if not already set

    uvicorn.run(app, port=8000)
