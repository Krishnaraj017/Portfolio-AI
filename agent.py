import os
from typing import Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from dotenv import load_dotenv

# LangGraph State

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str


# Context about Krishnaraj (from resume)
KRISHNARAJ_CONTEXT = os.getenv("KRISHNARAJ_CONTEXT")


class KrishnarajAgent:
    def __init__(self):
        self.llm = self._get_llm()
        self.graph = self._create_graph()
        self.session_memories: Dict[str, List[BaseMessage]] = {}
        self.max_memory_length = 20  # Keep last 20 messages (10 exchanges)

    def _get_llm(self):
        """Initialize Gemini LLM"""
        api_key = os.getenv(
            "GEMINI_API_KEY")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=0.7,
        )

    def _get_memory(self, session_id: str) -> List[BaseMessage]:
        """Get or create memory for a session"""
        if session_id not in self.session_memories:
            self.session_memories[session_id] = []
        return self.session_memories[session_id]

    def _update_memory(self, session_id: str, human_message: str, ai_message: str):
        """Update session memory with new messages"""
        if session_id not in self.session_memories:
            self.session_memories[session_id] = []

        # Add new messages
        self.session_memories[session_id].extend([
            HumanMessage(content=human_message),
            AIMessage(content=ai_message)
        ])

        # Keep only the last max_memory_length messages
        if len(self.session_memories[session_id]) > self.max_memory_length:
            self.session_memories[session_id] = self.session_memories[session_id][-self.max_memory_length:]

    def _create_prompt(self):
        """Create the chatbot prompt template"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions about Krishnaraj S, a Full Stack AI/ML Engineer. 
            
            Use the following context about Krishnaraj to answer questions accurately and helpfully:
            
            {context}
            
            Guidelines:
            1. Answer questions based on the provided context about Krishnaraj
            2. If asked about skills, experience, projects, or background, refer to the information provided
            3. Be friendly and professional in your responses
            4. If someone asks about contacting Krishnaraj, provide his email or LinkedIn
            5. If asked about something not in the context, politely mention you can only answer questions about Krishnaraj's professional background
            6. Keep responses concise but informative
            7. Use the conversation history to maintain context across messages
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{message}")
        ])

    def _chatbot_node(self, state: State) -> State:
        """LangGraph chatbot node"""
        prompt = self._create_prompt()

        # Get the last message
        last_message = state["messages"][-1]

        # Create chain
        chain = prompt | self.llm | StrOutputParser()

        # Get response
        response = chain.invoke({
            "context": state["context"],
            # All messages except the current one
            "chat_history": state["messages"][:-1],
            "message": last_message.content
        })

        # Add AI response to messages
        ai_message = AIMessage(content=response)

        return {
            "messages": state["messages"] + [ai_message],
            "context": state["context"]
        }

    def _create_graph(self):
        """Create LangGraph workflow"""
        workflow = StateGraph(State)
        workflow.add_node("chatbot", self._chatbot_node)
        workflow.set_entry_point("chatbot")
        workflow.set_finish_point("chatbot")
        return workflow.compile()

    def chat(self, message: str, session_id: str = "default") -> str:
        """Process a chat message and return response"""
        try:
            # Get memory for this session
            chat_history = self._get_memory(session_id)

            # Create initial state with current message and history
            messages = chat_history + [HumanMessage(content=message)]

            initial_state = {
                "messages": messages,
                "context": KRISHNARAJ_CONTEXT
            }

            # Run the graph
            result = self.graph.invoke(initial_state)

            # Get the AI response
            ai_response = result["messages"][-1].content

            # Update memory with the new exchange
            self._update_memory(session_id, message, ai_response)

            return ai_response

        except Exception as e:
            raise Exception(f"Error processing chat: {str(e)}")

    def get_active_sessions(self) -> Dict:
        """Get list of active sessions"""
        return {
            "active_sessions": list(self.session_memories.keys()),
            "total_sessions": len(self.session_memories)
        }

    def clear_session(self, session_id: str) -> bool:
        """Clear a specific session"""
        if session_id in self.session_memories:
            del self.session_memories[session_id]
            return True
        return False
