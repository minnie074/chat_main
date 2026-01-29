"""
LLM and Question Answering Module
Handles question answering using LLM with retrieval augmented generation
"""
from typing import List, Dict, Optional
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from config.settings import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    NOT_FOUND_RESPONSE, OPENAI_API_KEY,
    CONVERSATION_MEMORY_SIZE
)


class QuestionAnswerer:
    """Handles question answering using LLM and retrieved context"""
    
    def __init__(self, vector_store, use_openai: bool = True):
        """
        Initialize question answerer
        
        Args:
            vector_store: Vector store instance for retrieving context
            use_openai: Whether to use OpenAI (requires API key)
        """
        self.vector_store = vector_store
        self.use_openai = use_openai and OPENAI_API_KEY
        
        # Initialize conversation memory
        self.conversation_history: List[Dict] = []
        
        # System prompt for the LLM
        self.system_prompt = """You are a helpful AI assistant that answers questions based STRICTLY on the provided website content.

CRITICAL RULES:
1. ONLY use information from the provided context to answer questions
2. If the answer is not in the context, respond EXACTLY with: "The answer is not available on the provided website."
3. Do NOT use any external knowledge or information not present in the context
4. Do NOT make assumptions or inferences beyond what is explicitly stated
5. Always cite which part of the website your answer comes from when possible
6. Be concise and accurate

Context from website:
{context}

Conversation History:
{chat_history}

Question: {question}

Answer:"""
        
        # Initialize LLM
        if self.use_openai:
            self.llm = ChatOpenAI(
                model_name=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                openai_api_key=OPENAI_API_KEY
            )
            print(f"Initialized OpenAI LLM: {LLM_MODEL}")
        else:
            print("OpenAI API key not found. Will use retrieval-based responses only.")
    
    def answer_question(self, question: str) -> Dict:
        """
        Answer a question based on website content
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve relevant context
        relevant_docs = self.vector_store.search(question, top_k=4)
        
        # Check if we found any relevant documents
        if not relevant_docs:
            return {
                'answer': NOT_FOUND_RESPONSE,
                'sources': [],
                'confidence': 0.0
            }
        
        # Prepare context from retrieved documents
        context = self._prepare_context(relevant_docs)
        
        # Check if context is meaningful
        if not context or len(context.strip()) < 50:
            return {
                'answer': NOT_FOUND_RESPONSE,
                'sources': [],
                'confidence': 0.0
            }
        
        # Generate answer using LLM if available
        if self.use_openai:
            answer = self._generate_llm_answer(question, context)
        else:
            # Fallback to simple retrieval-based answer
            answer = self._generate_retrieval_answer(relevant_docs)
        
        # Extract sources
        sources = self._extract_sources(relevant_docs)
        
        # Calculate average confidence
        avg_confidence = sum(doc['similarity'] for doc in relevant_docs) / len(relevant_docs)
        
        # Store in conversation history
        self._add_to_history(question, answer)
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': avg_confidence,
            'num_sources': len(relevant_docs)
        }
    
    def _prepare_context(self, documents: List[Dict]) -> str:
        """
        Prepare context string from retrieved documents
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            text = doc['text']
            source = doc['metadata'].get('title', doc['metadata'].get('url', 'Unknown'))
            
            context_parts.append(f"[Source {i}: {source}]\n{text}\n")
        
        return "\n".join(context_parts)
    
    def _generate_llm_answer(self, question: str, context: str) -> str:
        """
        Generate answer using LLM
        
        Args:
            question: User's question
            context: Retrieved context
            
        Returns:
            Generated answer
        """
        try:
            # Prepare chat history for context
            chat_history = self._format_chat_history()
            
            # Create prompt
            prompt = self.system_prompt.format(
                context=context,
                chat_history=chat_history,
                question=question
            )
            
            # Generate response
            response = self.llm.predict(prompt)
            
            # Check if LLM says answer is not available
            if "not available" in response.lower() or "cannot answer" in response.lower():
                return NOT_FOUND_RESPONSE
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating LLM answer: {str(e)}")
            return self._generate_retrieval_answer([{'text': context}])
    
    def _generate_retrieval_answer(self, documents: List[Dict]) -> str:
        """
        Generate answer using simple retrieval (fallback)
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Answer based on most relevant document
        """
        if not documents:
            return NOT_FOUND_RESPONSE
        
        # Return the most relevant document's text
        most_relevant = documents[0]
        text = most_relevant['text']
        
        # Truncate if too long
        max_length = 500
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        source = most_relevant['metadata'].get('title', 'the website')
        return f"Based on {source}: {text}"
    
    def _extract_sources(self, documents: List[Dict]) -> List[Dict]:
        """
        Extract source information from documents
        
        Args:
            documents: Retrieved documents
            
        Returns:
            List of source dictionaries
        """
        sources = []
        seen_urls = set()
        
        for doc in documents:
            url = doc['metadata'].get('url', '')
            
            if url and url not in seen_urls:
                sources.append({
                    'url': url,
                    'title': doc['metadata'].get('title', 'Untitled'),
                    'similarity': doc.get('similarity', 0.0)
                })
                seen_urls.add(url)
        
        return sources
    
    def _format_chat_history(self) -> str:
        """
        Format conversation history for context
        
        Returns:
            Formatted chat history string
        """
        if not self.conversation_history:
            return "No previous conversation."
        
        # Only include recent history
        recent_history = self.conversation_history[-CONVERSATION_MEMORY_SIZE:]
        
        formatted = []
        for entry in recent_history:
            formatted.append(f"Human: {entry['question']}")
            formatted.append(f"Assistant: {entry['answer']}")
        
        return "\n".join(formatted)
    
    def _add_to_history(self, question: str, answer: str):
        """
        Add Q&A to conversation history
        
        Args:
            question: User's question
            answer: Bot's answer
        """
        self.conversation_history.append({
            'question': question,
            'answer': answer
        })
        
        # Keep only recent history to prevent context overflow
        if len(self.conversation_history) > CONVERSATION_MEMORY_SIZE * 2:
            self.conversation_history = self.conversation_history[-CONVERSATION_MEMORY_SIZE:]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_conversation_history(self) -> List[Dict]:
        """Get current conversation history"""
        return self.conversation_history


class ConversationalMemory:
    """
    Manages short-term conversational memory
    (Alternative implementation using LangChain's memory)
    """
    
    def __init__(self, memory_size: int = CONVERSATION_MEMORY_SIZE):
        """
        Initialize conversational memory
        
        Args:
            memory_size: Number of exchanges to remember
        """
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question",
            max_token_limit=memory_size * 100  # Approximate
        )
    
    def add_exchange(self, question: str, answer: str):
        """Add a question-answer exchange to memory"""
        self.memory.save_context(
            {"question": question},
            {"answer": answer}
        )
    
    def get_history(self) -> List:
        """Get conversation history"""
        return self.memory.load_memory_variables({})["chat_history"]
    
    def clear(self):
        """Clear memory"""
        self.memory.clear()