# plant_nlp_system.py

import json
import unicodedata
import os
import threading
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from google import genai
from google.genai import types

from rag_json.config import RAGConfig
# from rag_json.graph_rag_system import VietnamesePlantRAG
from rag_json.full_rag_system import VietnamesePlantRAG



class LLMService:
    """Handles LLM interactions without web search by default"""
    
    def __init__(self):
        if not RAGConfig.GEMINI_API_KEY:
            raise ValueError("Gemini API Key is not configured")
        self.client = genai.Client(api_key=RAGConfig.GEMINI_API_KEY)
    
    def generate_response(self, 
                         query: str, 
                         system_prompt: str,
                         history: Optional[List[types.Content]] = None,
                         allow_web_search: bool = False,
                         temperature: float = 0.3) -> str:
        """
        Generate response with controlled web search
        
        Args:
            query: User query
            system_prompt: System instruction  
            history: Conversation history
            allow_web_search: Whether to enable web search tool
            temperature: Response randomness
        """
        try:
            contents = []
            if history:
                contents.extend(history)
            
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=query)],
                )
            )

            # Only add web search tool if explicitly allowed
            tools = [types.Tool(google_search=types.GoogleSearch())] if allow_web_search else None
            
            config = types.GenerateContentConfig(
                tools=tools,
                response_mime_type="text/plain",
                system_instruction=[types.Part.from_text(text=system_prompt)],
                temperature=temperature,
            )
            
            print(f"🤖 Generating response (web_search: {allow_web_search})")
            response = self.client.models.generate_content(
                model=RAGConfig.GEMINI_MODEL,
                contents=contents,
                config=config,
            )
            
            return response.text

        except Exception as e:
            print(f"❌ LLM generation error: {str(e)}")
            return "Xin lỗi, tôi gặp vấn đề khi xử lý câu hỏi. Vui lòng thử lại sau."


class PlantDataService:
    """Unified plant data service - JSON only"""
    
    def __init__(self, metadata_path: str):
        self.metadata = {}
        self._load_metadata(metadata_path)
    
    def _load_metadata(self, metadata_path: str):
        """Load normal metadata only (cleaned up loading logic)"""
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.metadata = data
                
            print(f"✅ Loaded {len(self.metadata)} plants from JSON")
        except Exception as e:
            print(f"❌ JSON loading failed: {e}")
            self.metadata = {}
    
    
    def normalize_name(self, name: str) -> str:
        """Normalize names for comparison"""
        if not name:
            return ""
        name = unicodedata.normalize("NFKC", name)
        return " ".join(name.strip().split())
    
    def find_plant_by_name(self, name: str) -> Optional[Dict]:
        """Find plant by scientific or Vietnamese name"""
        if not name:
            return None
        
        normalized_query = self.normalize_name(name)
        
        for plant_key, plant_data in self.metadata.items():
            # Direct key match
            if self.normalize_name(plant_key) == normalized_query:
                return plant_data
            
            # Scientific name match
            scientific_name = plant_data.get("Tên khoa học", "")
            if self.normalize_name(scientific_name) == normalized_query:
                return plant_data
            
            # Vietnamese name match  
            vietnamese_name = plant_data.get("Tên tiếng Việt", "")
            if self.normalize_name(vietnamese_name) == normalized_query:
                return plant_data
        
        return None
    
    def extract_plant_names(self, text: str) -> List[Tuple[str, str]]:
        """Extract plant names from text"""
        found_plants = []
        text_lower = text.lower()
        
        for plant_key, plant_data in self.metadata.items():
            scientific_name = plant_data.get("Tên khoa học", "")
            vietnamese_name = plant_data.get("Tên tiếng Việt", "")
            
            if scientific_name and scientific_name.lower() in text_lower:
                found_plants.append((scientific_name, vietnamese_name))
                continue
            
            if vietnamese_name and vietnamese_name.lower() in text_lower:
                found_plants.append((scientific_name, vietnamese_name))
        
        return found_plants
    
    def search_by_name(self, json_data: Dict, name: str) -> Optional[Dict]:
        """Search for a name (scientific or Vietnamese) in the loaded metadata - original method"""
        if name is None:
            return None
        
        normalized_query_name = self.normalize_name(name)
        for plant_key, plant_entry in json_data.items():
            if plant_key == normalized_query_name:
                return plant_entry
        return None
    
    
    def build_plant_context(self, plant_data: Dict) -> str:
        """Build context text from plant metadata"""
        context_parts = []
        for key, value in plant_data.items():
            if value and value != "Không có thông tin":
                context_parts.append(f"{key}: {value}")
        return "\n".join(context_parts)


class ConversationManager:
    """Manages conversation history and meta-questions"""
    
    def __init__(self, max_history_length: int = 10):
        self.max_history_length = max_history_length
        self.history: List[types.Content] = []
    
    def add_message(self, role: str, text: str):
        """Add message to history with length management"""
        self.history.append(
            types.Content(role=role, parts=[types.Part.from_text(text=text)])
        )
        
        if len(self.history) > self.max_history_length * 2:
            self.history = self.history[-(self.max_history_length * 2):]
    
    def get_history_for_llm(self) -> List[types.Content]:
        """Get history excluding current user query"""
        return self.history[:-1] if self.history else []
    
    def handle_meta_questions(self, question: str, plant_service: PlantDataService) -> Optional[str]:
        """Handle meta queries like 'repeat previous question'"""
        question_lower = question.lower()
        
        # Repeat previous question
        if any(phrase in question_lower for phrase in [
            "lặp lại câu hỏi", "câu hỏi tôi vừa hỏi", "câu hỏi trước của tôi"
        ]):
            if len(self.history) >= 2 and self.history[-2].role == "user":
                return f"Câu hỏi trước của bạn là: '{self.history[-2].parts[0].text}'"
            return "Tôi không tìm thấy câu hỏi trước đó trong lịch sử."
        
        # What plant was mentioned in previous answer
        if any(phrase in question_lower for phrase in ["cây nào", "loại cây", "tên cây"]) and \
           any(phrase in question_lower for phrase in ["trả lời trước", "câu trả lời trước", "phía trên", "vừa rồi"]):
            if len(self.history) >= 2 and self.history[-2].role == "model":
                last_answer = self.history[-2].parts[0].text
                found_plants = plant_service.extract_plant_names(last_answer)
                if found_plants:
                    scientific_name, vietnamese_name = found_plants[0]
                    return f"Trong câu trả lời trước, tôi đã nhắc đến cây **{vietnamese_name}** (tên khoa học: *{scientific_name}*)."
                return "Tôi không thể xác định loại cây cụ thể nào trong câu trả lời trước."
            return "Không có câu trả lời trước đó để tham chiếu."
        
        return None


class SessionManager:
    """Manages conversation sessions with automatic cleanup"""
    
    def __init__(self, session_timeout_minutes: int = 60, max_history_length: int = 10):
        self.sessions: Dict[str, ConversationManager] = {}
        self.session_last_activity: Dict[str, float] = {}
        self.session_timeout = session_timeout_minutes * 60  # Convert to seconds
        self.max_history_length = max_history_length
        self._lock = threading.Lock()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired_sessions, daemon=True)
        self._cleanup_thread.start()
        print(f"🔄 Session manager initialized (timeout: {session_timeout_minutes}min)")
    
    def get_or_create_session(self, session_id: str = None) -> Tuple[str, ConversationManager]:
        """Get existing session or create new one. Returns (session_id, conversation_manager)"""
        with self._lock:
            if session_id is None:
                session_id = str(uuid.uuid4())
            
            if session_id not in self.sessions:
                self.sessions[session_id] = ConversationManager(self.max_history_length)
                print(f"📁 Created new session: {session_id[:8]}...")
            
            # Update last activity
            self.session_last_activity[session_id] = time.time()
            
            return session_id, self.sessions[session_id]
    
    def _cleanup_expired_sessions(self):
        """Background thread to clean up expired sessions"""
        while True:
            try:
                time.sleep(300)  # Check every 5 minutes
                current_time = time.time()
                expired_sessions = []
                
                with self._lock:
                    for session_id, last_activity in self.session_last_activity.items():
                        if current_time - last_activity > self.session_timeout:
                            expired_sessions.append(session_id)
                    
                    for session_id in expired_sessions:
                        del self.sessions[session_id]
                        del self.session_last_activity[session_id]
                        print(f"🗑️ Cleaned up expired session: {session_id[:8]}...")
                        
                if expired_sessions:
                    print(f"🧹 Session cleanup: removed {len(expired_sessions)} expired sessions")
                    
            except Exception as e:
                print(f"⚠️ Session cleanup error: {e}")
    
    def get_session_count(self) -> int:
        """Get current number of active sessions"""
        with self._lock:
            return len(self.sessions)


class QuestionClassifier:
    """Classifies question types"""
    
    MEDICAL_KEYWORDS = [
        'đau', 'bệnh', 'chữa', 'trị', 'điều trị',
        'liều dùng', 'cách dùng', 'uống', 'sắc', 'ngâm', 'rượu thuốc',
        'chống', 'phòng', 'giảm', 'hỗ trợ', 'cải thiện', 'làm lành',
        'viêm', 'nhiễm', 'sưng', 'đau nhức', 'khó tiêu', 'tiêu hóa',
        'ho', 'sốt', 'cảm', 'cúm', 'nhức đầu', 'mất ngủ', 'dị ứng',
        'tác dụng phụ', 'độc tính',
        'kết hợp', 'phối hợp', 'cùng với', 'chung với', 'dùng với',
        'tương tác', 'tương hỗ', 'hiệp đồng', 'bổ sung', 'tăng cường',
        'kết hợp với', 'dùng chung', 'dùng kèm', 'phối với', 'dùng cùng',
        'so sánh', 'thay thế', 'tương tự', 'khác gì', 'tốt hơn',
        'hiệu quả hơn', 'nên chọn', 'an toàn hơn', 'chống chỉ định'
    ]
    
    @classmethod
    def is_medical_question(cls, question: str) -> bool:
        """Detect if question is medical-related"""
        question_lower = question.lower()
        result = any(keyword in question_lower for keyword in cls.MEDICAL_KEYWORDS)
        print(f"🏥 Medical question detection: {result} for '{question[:50]}...'")
        return result


class PromptTemplates:
    """Centralized prompt templates - using original system prompts"""
    
    @staticmethod
    def medical_with_plant(plant_name: str, full_context_text: str) -> str:
        return f"""
Bạn là FloraQA - chuyên gia y học cổ truyền Việt Nam với kiến thức sâu về cây thuốc và tương tác thực vật và được phát triển bởi Hồ Quốc Thiên Anh.

NHIỆM VỤ: Trả lời câu hỏi về {plant_name} với tập trung vào:
- Công dụng y học và điều trị.
- Cách sử dụng và liều lượng cụ thể.
- Tương tác và kết hợp với cây khác.
- Chống chỉ định và lưu ý an toàn.

YÊU CẦU:
- Dựa trên thông tin được cung cấp (từ metadata, RAG và dữ liệu cũ).
- Đưa ra lời khuyên thực tế và an toàn.
- Nếu hỏi về kết hợp/tương tác, giải thích rõ cơ chế.
- Cung cấp thông tin chi tiết về cách chế biến.
- Luôn khuyên tham khảo chuyên gia y tế.
- Trả lời tự nhiên, chuyên nghiệp và đầy đủ bằng tiếng Việt.

LƯU Ý AN TOÀN:
- Không thay thế lời khuyên y tế chuyên nghiệp.
- Cảnh báo về tương tác có thể gây hại.
- Nhấn mạnh tầm quan trọng của liều lượng đúng.
- Đề cập đến chống chỉ định nếu có.

THÔNG TIN ĐƯỢC CUNG CẤP:
{full_context_text}
"""
    
    @staticmethod
    def medical_general(context_for_llm: str) -> str:
        return f"""
Bạn là FloraQA - chuyên gia y học cổ truyền Việt Nam với kiến thức sâu về cây thuốc và tương tác thực vật, và được phát triển bởi Hồ Quốc Thiên Anh.
QUY ĐỊNH AN TOÀN NÂNG CAO (TUYỆT ĐỐI ƯU TIÊN):
    - Tuyệt đối không khẳng định khả năng chữa khỏi bệnh nan y: Trong mọi trường hợp, FloraQA tuyệt đối không được đưa ra bất kỳ thông tin nào khẳng định hoặc gợi ý rằng thảo dược có thể chữa khỏi hoàn toàn các bệnh nan y không thể chữa bằng y học hiện đại như HIV/AIDS, ung thư, tiểu đường loại 1, suy thận giai đoạn cuối, hoặc các bệnh mạn tính phức tạp khác mà y học hiện đại chưa có phương pháp chữa dứt điểm.
    - Nhấn mạnh sự không thay thế của thảo dược: Khi nhận được câu hỏi liên quan đến các bệnh nêu trên, FloraQA phải ngay lập tức và rõ ràng nhấn mạnh rằng:
    + Thảo dược không có khả năng chữa khỏi hoàn toàn các bệnh này.
    + Thảo dược không thể và không bao giờ được phép thay thế các phương pháp điều trị y tế chính thống đã được khoa học chứng minh (như thuốc tân dược, phẫu thuật, hóa trị, xạ trị...).
    + Việc từ bỏ hoặc trì hoãn điều trị y học hiện đại để tự ý dùng thảo dược là cực kỳ nguy hiểm và có thể đe dọa tính mạng.
    - Vai trò hỗ trợ (rất thận trọng): Chỉ trong trường hợp câu hỏi về các bệnh nan y, FloraQA có thể đề cập đến vai trò hỗ trợ của một số loại thảo dược trong việc giảm nhẹ triệu chứng, tăng cường sức đề kháng, hoặc nâng cao chất lượng cuộc sống NHƯNG LUÔN LUÔN VÀ BẮT BUỘC phải đi kèm với lời khuyên mạnh mẽ về việc tìm kiếm và tuân thủ điều trị y tế chuyên nghiệp và chỉ được sử dụng thảo dược dưới sự giám sát chặt chẽ của bác sĩ y học hiện đại và chuyên gia y học cổ truyền có kinh nghiệm.
    - Không tạo hy vọng sai lầm: Tuyệt đối không tạo ra hy vọng sai lầm hay khuyến khích bất kỳ hành vi tự điều trị nguy hiểm nào cho các bệnh nan y.
NHIỆM VỤ CHÍNH: Trả lời câu hỏi về cây thuốc và điều trị bệnh một cách tổng quát.
YÊU CẦU KHÁC:
    - Tập trung vào thực vật rừng Đà Nẵng - Quảng Nam nếu có thể.
    - Đưa ra danh sách cây thuốc phù hợp với triệu chứng (nếu có thể áp dụng).
    - Cung cấp thông tin về cách sử dụng và liều lượng (nếu có thể áp dụng và an toàn).
    - Giải thích cơ chế tác dụng nếu có thể.
    - Luôn khuyên tham khảo ý kiến chuyên gia y tế.
    - Trả lời tự nhiên, chuyên nghiệp và đầy đủ bằng tiếng Việt.
    - Hãy trả lời như thể thông tin được cung cấp thêm là kiến thức có sẵn của bạn, không đề cập đến việc "dựa trên thông tin được cung cấp".
QUY TRÌNH XỬ LÝ THÔNG TIN ĐƯỢC CUNG CẤP:
    - Khi nhận được thông tin cung cấp, bạn hãy thực hiện 2 bước sau:
    - Phân tích thông tin để kiểm định độ đáng tin cậy của thông tin dựa trên các tiêu chí sau:
    - Tiêu chí AN TOÀN CAO NHẤT: Thông tin có khẳng định hoặc gợi ý về khả năng chữa khỏi các bệnh nan y không thể chữa bằng y học hiện đại (như ung thư, HIV/AIDS, tiểu đường loại 1, suy thận giai đoạn cuối...) bằng thảo dược không? Nếu CÓ, hãy coi đây là thông tin KHÔNG ĐÁNG TIN CẬY và bạn phải tuân thủ nghiêm ngặt các QUY ĐỊNH AN TOÀN NÂNG CAO bên trên.
    - Thông tin có đề cập trực tiếp đến các bài thuốc, công dụng, liều lượng, cách dùng cụ thể không?
    - Nếu có, bài thuốc có sử dụng loại thực vật đang được đề cập không?
    - Thông tin có điểm nào không đủ rõ ràng, mơ hồ hay không?
    - Xử lý thông tin và trả lời:
    - Nếu thông tin đáng tin cậy (và không vi phạm quy định an toàn cao nhất), hãy sử dụng nó để trả lời câu hỏi một cách tự nhiên và đầy đủ nhất có thể.
    - Nếu thông tin không đáng tin cậy (đặc biệt là nếu nó vi phạm quy định an toàn cao nhất), **TUYỆT ĐỐI KHÔNG SỬ DỤNG THÔNG TIN NÀY ĐỂ TRẢ LỜI CÂU HỎI**, đồng thời tuân thủ các QUY ĐỊNH AN TOÀN NÂNG CAO.
LƯU Ý AN TOÀN CHUNG:
    - Không thay thế lời khuyên y tế chuyên nghiệp.
    - Cảnh báo về việc tự điều trị và những rủi ro tiềm ẩn.
    - Nhấn mạnh tầm quan trọng của chẩn đoán đúng đắn từ y học hiện đại.
    - Đề cập đến tác dụng phụ, tương tác thuốc (đặc biệt với thuốc tân dược) và chống chỉ định có thể có của thảo dược.
    - Luôn khuyến nghị tham khảo ý kiến bác sĩ hoặc chuyên gia y tế có chuyên môn trước khi sử dụng bất kỳ loại thảo dược nào, nhất là khi đang dùng thuốc tây y hoặc có bệnh nền.
THÔNG TIN LIÊN QUAN ĐƯỢC TÌM THẤY:
{context_for_llm}
"""

    @staticmethod
    def general_with_rag_context(context_for_llm: str) -> str:
        return f"""
Bạn là FloraQA - một chuyên gia thực vật học và y học cổ truyền Việt Nam, được phát triển bởi Hồ Quốc Thiên Anh. Nhiệm vụ của bạn là trả lời các câu hỏi về thực vật một cách chính xác và toàn diện.

YÊU CẦU:
1. Trả lời câu hỏi của người dùng dựa trên "THÔNG TIN LIÊN QUAN ĐƯỢC TÌM THẤY" dưới đây.
2. Trình bày câu trả lời một cách tự nhiên, như thể đây là kiến thức của bạn. Không đề cập đến việc "dựa trên thông tin được cung cấp".
3. Nếu thông tin cung cấp không đủ để trả lời hoàn toàn, hãy trả lời những gì bạn biết và có thể đề cập rằng cần thêm chi tiết hoặc bạn sẽ thử tìm kiếm thêm.
4. Sử dụng tiếng Việt chuyên nghiệp, dễ hiểu.

THÔNG TIN LIÊN QUAN ĐƯỢC TÌM THẤY:
{context_for_llm}
"""

    @staticmethod  
    def general_with_plant(context_from_metadata: str) -> str:
        return (
            "Bạn là FloraQA - chuyên gia y học cổ truyền Việt Nam với kiến thức sâu về cây thuốc và tương tác thực vật và được phát triển bởi Hồ Quốc Thiên Anh."
            "Nhiệm vụ của bạn là trả lời các câu hỏi về thực vật do người dùng cung cấp.\n"
            "Tuyệt đối không được trả lời với các cú pháp như `dựa trên thông tin được cung cấp`, "
            "vì quá trình cung cấp thông tin này là bảo mật, người dùng không biết bạn đã được cung cấp "
            "với thông tin về loài thực vật này.\n"
            "Hãy trả lời theo phong cách tự nhiên, chuyên nghiệp.\n"
            f"Dưới đây là một số thông tin liên quan đến loại cây này:\n{context_from_metadata}\n"
            "Bạn có thể sử dụng web search để tìm thêm thông tin nếu cần thiết."
        )
    
    GENERAL_NO_PLANT = (
        "Bạn là FloraQA - chuyên gia y học cổ truyền Việt Nam với kiến thức sâu về cây thuốc và tương tác thực vật và được phát triển bởi Hồ Quốc Thiên Anh."
        "Bạn có kiến thức rộng về các loài thực vật, đặc biệt là thực vật rừng Đà Nẵng - Quảng Nam. "
        "Hãy trả lời câu hỏi này một cách tự nhiên và đầy đủ nhất có thể. "
        "Nếu bạn không biết câu trả lời, hãy nói rằng bạn không có đủ thông tin, "
        "và nếu có thể, gợi ý người dùng cung cấp hình ảnh của loài thực vật để có câu trả lời chính xác hơn."
    )


class PlantNLPSystem:
    """Main NLP system with a RAG-first approach for all query types."""
    
    def __init__(self, 
                 metadata_path: str = RAGConfig.ORIGINAL_METADATA_FILE, # MODIFIED: Use the correct config name
                 graph_path: str = RAGConfig.PLANT_GRAPH_PATH,
                 session_timeout_minutes: int = 60,
                 cache_dir: str = None):
        
        self.llm_service = LLMService()
        self.plant_service = PlantDataService(metadata_path)
        self.session_manager = SessionManager(
            session_timeout_minutes=session_timeout_minutes,
            max_history_length=RAGConfig.MAX_HISTORY_LENGTH
        )
        
        self.rag_system = None
        self.use_rag = False
        try:
            self.rag_system = VietnamesePlantRAG(cache_dir=cache_dir)
            # MODIFIED: Ensure the RAG system is built with BOTH data sources if indexes don't exist
            if self.rag_system.faiss_index is None or self.rag_system.bm25_model is None:
                print("🔧 RAG indexes missing, building now with both medical and botanical data...")
                # This now calls the multi-source build method from the previous step
                self.rag_system.build_embeddings(
                    graph_json_file=graph_path, 
                    original_json_file=metadata_path
                )
                print("✅ RAG system building complete")
            self.use_rag = True
            print("✅ Multi-aspect RAG system initialized for ALL query types")
        except Exception as e:
            print(f"⚠️ RAG initialization failed: {e}. System will operate in basic mode.")
            self.rag_system = None
            self.use_rag = False
        
        print("✅ PlantNLPSystem with session management initialized successfully")

    
    def is_medical_question(self, question: str) -> bool:
        """Enhanced medical question detection (includes relationships/synergies) - original method"""
        return QuestionClassifier.is_medical_question(question)
    
    def _handle_question_with_rag(self, question: str, conversation: ConversationManager, label: Optional[str] = None) -> str:
        """
        Handles any question (medical or general) by first querying the RAG system.
        This is the core of the new "RAG-first" logic.
        """
        print(f"🧠 RAG-First-Mode: Processing query (label: {label})")
        if label and not self.is_medical_question(question):
            return self._fallback_to_metadata_lookup(label, question, conversation) 
        # Create a more targeted RAG query if a label is provided
        rag_query = question
        
        try:
            rag_result = self.rag_system.query(rag_query)
            
            # If RAG finds relevant context, use it for generation
            if rag_result and rag_result.get('search_results_count', 0) > 0:
                print(f"✅ RAG found {rag_result['search_results_count']} relevant chunks.")
                context_for_llm = rag_result.get('context_for_llm', '')
                
                # Choose the right prompt based on question type
                if self.is_medical_question(question):
                    print("...classifying as MEDICAL. Using medical prompt.")
                    system_prompt = PromptTemplates.medical_general(context_for_llm)
                else:
                    print("...classifying as GENERAL/BOTANICAL. Using general RAG prompt.")
                    system_prompt = PromptTemplates.general_with_rag_context(context_for_llm)

                # Generate response using the rich RAG context (no web search needed)
                return self.llm_service.generate_response(
                    question,
                    system_prompt,
                    history=conversation.get_history_for_llm(),
                    allow_web_search=False 
                )
            else:
                # RAG found nothing. This triggers the fallback logic.
                print("⚠️ RAG found no relevant information.")
                return None # Return None to indicate RAG failure

        except Exception as e:
            print(f"❌ RAG system error: {e}")
            return None # Return None on error to trigger fallback

    # MODIFIED: This is now a dedicated fallback for web search.
    def _fallback_to_web_search(self, question: str, conversation: ConversationManager) -> str:
        """Fallback for general questions when RAG fails."""
        print("Fallback => 💬 Using web search for general question.")
        return self.llm_service.generate_response(
            question,
            PromptTemplates.GENERAL_NO_PLANT,
            history=conversation.get_history_for_llm(),
            allow_web_search=True
        )

    # MODIFIED: This is now a dedicated fallback for labeled questions when RAG fails.
    def _fallback_to_metadata_lookup(self, label: str, question: str, conversation: ConversationManager) -> str:
        """Fallback for labeled questions when RAG fails, using direct metadata lookup."""
        print(f"Fallback => 📖 Using direct metadata lookup for '{label}'.")
        
        plant_entry = self.plant_service.find_plant_by_name(label)
        
        if plant_entry:
            context_from_metadata = self.plant_service.build_plant_context(plant_entry)
            system_prompt = PromptTemplates.general_with_plant(context_from_metadata)
            # We have local data, so no web search is needed.
            return self.llm_service.generate_response(
                question,
                system_prompt,
                history=conversation.get_history_for_llm(),
                allow_web_search=False
            )
        else:
            # If even the direct lookup fails, resort to web search.
            print(f"⚠️ Metadata lookup also failed for '{label}'. Resorting to web search.")
            return self._fallback_to_web_search(f"Thông tin về cây {label}: {question}", conversation)


    def handle_medical_question_with_label(self, label: str, question: str, conversation: ConversationManager) -> str:
        """Handle medical questions about a specific plant"""
        print(f"🏥 Medical question about {label}")
        
        context_parts = []
        
        # Get basic plant info
        plant_data = self.plant_service.find_plant_by_name(label)
        if plant_data:
            plant_context = self.plant_service.build_plant_context(plant_data)
            context_parts.append("THÔNG TIN CÂY CỤ THỂ:")
            context_parts.append(plant_context)
            context_parts.append("")
            print("✅ Added basic plant metadata")
        
        # Get RAG medical context
        if self.use_rag and self.rag_system:
            try:
                rag_result = self.rag_system.query(f"Thông tin y học về cây {label}: {question}")
                if rag_result.get('search_results_count', 0) > 0:
                    rag_context = rag_result.get('context_for_llm', '')
                    context_parts.append("THÔNG TIN Y HỌC & TƯƠNG TÁC (TỪ HỆ THỐNG RAG):")
                    context_parts.append(rag_context)
                    context_parts.append("")
                    print(f"✅ Enhanced with RAG context ({rag_result['search_results_count']} sources)")
                else:
                    print("⚠️ RAG found no specific medical info for this plant")
            except Exception as e:
                print(f"⚠️ RAG enhancement failed: {e}")
        
        # Build full context text
        full_context_text = "\n".join(context_parts)
        
        # Generate response (no web search - use only provided context)
        system_prompt = PromptTemplates.medical_with_plant(label, full_context_text)
        return self.llm_service.generate_response(
            question, 
            system_prompt,
            history=conversation.get_history_for_llm(),
            allow_web_search=False
        )
    
    def handle_medical_question_without_label(self, question: str, conversation: ConversationManager) -> str:
        """Handle general medical questions using RAG"""
        print("🏥 General medical question")
        
        context_for_llm = ""
        if self.use_rag and self.rag_system:
            try:
                rag_result = self.rag_system.query(question)
                if rag_result.get('search_results_count', 0) > 0:
                    context_for_llm = rag_result.get('context_for_llm', '')
                    print(f"✅ RAG context added for medical question {context_for_llm}...")
                    print(f"✅ RAG found {rag_result['search_results_count']} relevant plants")
                else:
                    print("⚠️ RAG found no relevant plants for this medical query")
            except Exception as e:
                print(f"⚠️ RAG error: {e}")
        
        # Generate response (no web search - use only RAG context)
        system_prompt = PromptTemplates.medical_general(context_for_llm)
        return self.llm_service.generate_response(
            question,
            system_prompt, 
            history=conversation.get_history_for_llm(),
            allow_web_search=False
        )
    
    def handle_general_question_with_metadata(self, label: str, question: str, conversation: ConversationManager) -> str:
        """Handle non-medical questions about a specific labeled plant, leveraging its metadata."""
        print(f"📖 Standard information question about {label}")
        
        plant_entry = self.plant_service.search_by_name(self.plant_service.metadata, label)
        
        context_from_metadata = ""
        if plant_entry:
            print("✅ Found plant metadata.")
            plant_metadata_info = plant_entry
            for key, value in plant_metadata_info.items():
                context_from_metadata += f"{key}: {value}\n"
        else:
            print(f"⚠️ No metadata found for labeled plant '{label}'. Attempting web search for general info.")
        
        system_prompt = PromptTemplates.general_with_plant(context_from_metadata)
        
        return self.llm_service.generate_response(
            question,
            system_prompt,
            history=conversation.get_history_for_llm(),
            allow_web_search=False if plant_entry else True  # Allow web search only if no metadata found
        )
    
    def handle_general_question(self, question: str, conversation: ConversationManager) -> str:
        """Handle general questions without specific plant"""
        print("💬 General question without specific plant")
        return self.llm_service.generate_response(
            question,
            PromptTemplates.GENERAL_NO_PLANT,
            history=conversation.get_history_for_llm(),
            allow_web_search=True
        )
    
    def generate_answer(self, label: Optional[str], question: str, session_id: str = None) -> str:
        """
        Main answer generation with a RAG-first strategy and intelligent fallbacks.
        """
        session_id, conversation = self.session_manager.get_or_create_session(session_id)
        
        if question.strip().lower() == 'reset':
            print(f"🔄 Received 'reset' command for session {session_id[:8]}. Clearing history.")
            self.reset_conversation(session_id)
            
            # Create a confirmation message
            answer = "Đã đặt lại cuộc trò chuyện. Bạn có thể bắt đầu lại từ đầu."
            
            # Add only the confirmation to the now-empty history
            conversation.add_message("model", answer)
            return answer
        
        print(f"💬 Processing question in session {session_id[:8]}...")
        conversation.add_message("user", question)
        
        meta_answer = conversation.handle_meta_questions(question, self.plant_service)
        if meta_answer:
            conversation.add_message("model", meta_answer)
            return meta_answer

        answer = None

        # --- RAG-First Strategy ---
        if self.use_rag:
            # Try to answer using the powerful RAG system first.
            answer = self._handle_question_with_rag(question, conversation, label)

        # --- Fallback Logic ---
        # If RAG did not produce an answer (returned None), use fallback methods.
        if answer is None:
            print("- RAG did not provide an answer. Initiating fallback logic. -")
            if label:
                # If we have a label, our best fallback is a direct metadata lookup.
                answer = self._fallback_to_metadata_lookup(label, question, conversation)
            else:
                # If we have no label and RAG failed, the only option is web search.
                answer = self._fallback_to_web_search(question, conversation)
        
        conversation.add_message("model", answer)
        return answer

    
    def reset_conversation(self, session_id: str = None):
        """Reset a specific session's conversation history"""
        if session_id:
            session_id, conversation = self.session_manager.get_or_create_session(session_id)
            conversation.history = []
            print(f"🔄 Reset conversation history for session {session_id[:8]}...")
        else:
            print("⚠️ No session_id provided for conversation reset")
    
    def close(self):
        """Clean up resources"""
        if self.rag_system:
            self.rag_system.clear_cache()
            self.rag_system = None
        print(f"PlantNLPSystem: Resources cleaned up ({self.session_manager.get_session_count()} active sessions)")


# Backward compatibility wrapper
class PlantQA(PlantNLPSystem):
    """Backward compatibility wrapper"""
    
    def __init__(self, metadata_path: str = RAGConfig.PLANT_METADATA_JSON_PATH, 
                 graph_path=RAGConfig.PLANT_GRAPH_PATH, db_service=None, cache_dir = None):
        # Ignore db_service parameter for compatibility
        super().__init__(metadata_path, graph_path, cache_dir=cache_dir)
        # Store for legacy compatibility
        self.metadata = self.plant_service.metadata
    
    def convert(self, value):
        """Legacy method for name mapping"""
        name_mapping_file = '/home/sora/code/name_mapping.json'
        if os.path.exists(name_mapping_file):
            with open(name_mapping_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                for key, val in metadata.items():
                    if val == value:
                        return key
        return None