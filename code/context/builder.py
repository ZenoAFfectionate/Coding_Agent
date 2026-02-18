"""ContextBuilder - GSSC流水线实现

实现 Gather-Select-Structure-Compress 上下文构建流程：
1. Gather: 从多源收集候选信息（历史、记忆、RAG、工具结果）
2. Select: 基于优先级、相关性、多样性筛选
3. Structure: 组织成结构化上下文模板
4. Compress: 在预算内压缩与规范化
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import tiktoken
import math

from ..core.message import Message
from ..tools import MemoryTool, RAGTool


@dataclass
class ContextPacket:
    """Context Information Packet"""
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    relevance_score: float = 0.0  # 0.0-1.0
    
    def __post_init__(self):
        """compute token number automatically"""
        if self.token_count == 0:
            self.token_count = count_tokens(self.content)


@dataclass
class ContextConfig:
    """Context Config"""
    max_tokens: int = 8000       # total token budget for context
    reserve_ratio: float = 0.15  # generation reserve (10-20%)
    min_relevance: float = 0.3   # minimum relevance threshold
    enable_mmr: bool = True      # enable Maximal Marginal Relevance (diversity)
    mmr_lambda: float = 0.7      # MMR balance parameter (0=pure diversity, 1=pure relevance)
    system_prompt_template: str = ""  # system prompt template
    enable_compression: bool = True   # enbale context compress
    
    def get_available_tokens(self) -> int:
        """get available tokens with reserve deduction"""
        return int(self.max_tokens * (1 - self.reserve_ratio))


class ContextBuilder:
    """Context Builder with GSSC Pipeline
    
    Example：
    ```python
    builder = ContextBuilder(
        memory_tool=memory_tool,
        rag_tool=rag_tool,
        config=ContextConfig(max_tokens=8000)
    )
    
    context = builder.build(
        user_query="用户问题",
        conversation_history=[...],
        system_instructions="系统指令"
    )
    ```
    """
    
    def __init__(
        self,
        memory_tool: Optional[MemoryTool] = None,
        rag_tool: Optional[RAGTool] = None,
        config: Optional[ContextConfig] = None
    ):
        self.memory_tool = memory_tool
        self.rag_tool = rag_tool
        self.config = config or ContextConfig()
        self._encoding = tiktoken.get_encoding("cl100k_base")
    
    def build(
        self,
        user_query: str,
        conversation_history: Optional[List[Message]] = None,
        system_instructions: Optional[str] = None,
        additional_packets: Optional[List[ContextPacket]] = None
    ) -> str:
        """Create complete context, return structured context"""
        # 1. Gather: gather candidate information
        packets = self._gather(
            user_query=user_query,
            conversation_history=conversation_history or [],
            system_instructions=system_instructions,
            additional_packets=additional_packets or []
        )
        
        # 2. Select: select and perform sorting
        selected_packets = self._select(packets, user_query)
        
        # 3. Structure: organize to structured template
        structured_context = self._structure(
            selected_packets=selected_packets,
            user_query=user_query,
            system_instructions=system_instructions
        )
        
        # 4. Compress: compress context if exceed
        final_context = self._compress(structured_context)
        
        return final_context
    
    def _gather(
        self,
        user_query: str,
        conversation_history: List[Message],
        system_instructions: Optional[str],
        additional_packets: List[ContextPacket]
    ) -> List[ContextPacket]:
        """gather candidate information"""
        packets = []
        
        # P0: system instruction with strong constraint
        if system_instructions:
            packets.append(ContextPacket(
                content=system_instructions,
                metadata={"type": "instructions"}
            ))
        
        # P1: get key conclusion from memory tool
        if self.memory_tool:
            try:
                # search relevant memory with task state
                state_results = self.memory_tool.execute(
                    "search",
                    query="(任务状态 OR 子目标 OR 结论 OR 阻塞)",
                    min_importance=0.7,
                    limit=5
                )
                if state_results and "未找到" not in state_results:
                    packets.append(ContextPacket(
                        content=state_results,
                        metadata={"type": "task_state", "importance": "high"}
                    ))
                
                # search relevant memroy with user query
                related_results = self.memory_tool.execute(
                    "search",
                    query=user_query,
                    limit=5
                )
                if related_results and "未找到" not in related_results:
                    packets.append(ContextPacket(
                        content=related_results,
                        metadata={"type": "related_memory"}
                    ))
            except Exception as e:
                print(f"⚠️ 记忆检索失败: {e}")
        
        # P2: get fact from RAG tool
        if self.rag_tool:
            try:
                rag_results = self.rag_tool.run({
                    "action": "search",
                    "query": user_query,
                    "limit": 5
                })
                if rag_results and "未找到" not in rag_results and "错误" not in rag_results:
                    packets.append(ContextPacket(
                        content=rag_results,
                        metadata={"type": "knowledge_base"}
                    ))
            except Exception as e:
                print(f"⚠️ RAG failed: {e}")
        
        # P3: chat history for supplmentary
        if conversation_history:
            # only keep N most recently
            recent_history = conversation_history[-10:]
            history_text = "\n".join([
                f"[{msg.role}] {msg.content}"
                for msg in recent_history
            ])
            packets.append(ContextPacket(
                content=history_text,
                metadata={"type": "history", "count": len(recent_history)}
            ))
        
        # extend addition packets (e.g. tool results, retrievals)
        packets.extend(additional_packets)
        
        return packets
    
    def _select(
        self,
        packets: List[ContextPacket],
        user_query: str
    ) -> List[ContextPacket]:
        """Select based on score and budget"""
        # 1) compute relevance with keyword overlapped
        query_tokens = set(user_query.lower().split())
        for packet in packets:
            content_tokens = set(packet.content.lower().split())
            if len(query_tokens) > 0:
                overlap = len(query_tokens & content_tokens)
                packet.relevance_score = overlap / len(query_tokens)
            else:
                packet.relevance_score = 0.0
        
        # 2) compute recency score (exponential decay)
        def recency_score(ts: datetime) -> float:
            delta = max((datetime.now() - ts).total_seconds(), 0)
            tau = 3600  # an hour time constant
            return math.exp(-delta / tau)
        
        # 3) compute compound score：70% relevance and 30% recency
        scored_packets: List[Tuple[float, ContextPacket]] = []
        for p in packets:
            rec = recency_score(p.timestamp)
            score = 0.7 * p.relevance_score + 0.3 * rec
            scored_packets.append((score, p))
        
        # 4) take system instruction out and consider
        system_packets = [p for (_, p) in scored_packets if p.metadata.get("type") == "instructions"]
        remaining = [p for (s, p) in sorted(scored_packets, key=lambda x: x[0], reverse=True)
                     if p.metadata.get("type") != "instructions"]
        
        # 5) filter based on min_relevance
        filtered = [p for p in remaining if p.relevance_score >= self.config.min_relevance]
        
        # 6) padding according to budget
        available_tokens = self.config.get_available_tokens()
        selected: List[ContextPacket] = []
        used_tokens = 0
        
        # put system instruction without sorting
        for p in system_packets:
            if used_tokens + p.token_count <= available_tokens:
                selected.append(p)
                used_tokens += p.token_count
        
        # put the rest based on score
        for p in filtered:
            if used_tokens + p.token_count > available_tokens:
                continue
            selected.append(p)
            used_tokens += p.token_count
        
        return selected
    
    def _structure(
        self,
        selected_packets: List[ContextPacket],
        user_query: str,
        system_instructions: Optional[str]
    ) -> str:
        """Organize to structured context template"""
        sections = []
        
        # [Role & Policies] - system instruction and agent role
        p0_packets = [p for p in selected_packets if p.metadata.get("type") == "instructions"]
        if p0_packets:
            role_section = "[Role & Policies]\n"
            role_section += "\n".join([p.content for p in p0_packets])
            sections.append(role_section)
        
        # [Task] - current task
        sections.append(f"[Task]\nUser query：{user_query}")
        
        # [State] - Task state and key findings
        p1_packets = [p for p in selected_packets if p.metadata.get("type") == "task_state"]
        if p1_packets:
            state_section = "[State]\nKey improvements and undecided problems:\n"
            state_section += "\n".join([p.content for p in p1_packets])
            sections.append(state_section)
        
        # [Evidence] - evidence fact
        p2_packets = [
            p for p in selected_packets
            if p.metadata.get("type") in {"related_memory", "knowledge_base", "retrieval", "tool_result"}
        ]
        if p2_packets:
            evidence_section = "[Evidence]\nFact and Reference：\n"
            for p in p2_packets:
                evidence_section += f"\n{p.content}\n"
            sections.append(evidence_section)
        
        # [Context] - supplementary context
        p3_packets = [p for p in selected_packets if p.metadata.get("type") == "history"]
        if p3_packets:
            context_section = "[Context]\nChat history and background：\n"
            context_section += "\n".join([p.content for p in p3_packets])
            sections.append(context_section)
        
        # [Output] - constraint and format requirement
        output_section = """[Output]
                            Please respond in the following format:
                            1. Conclusion (concise and clear)
                            2. Evidence (list supporting evidence and sources)
                            3. Risks and Assumptions (if any)
                            4. Suggested Next Steps (if applicable)"""
        sections.append(output_section)
        
        return "\n\n".join(sections)
    
    def _compress(self, context: str) -> str:
        """Compress and normalize context"""
        if not self.config.enable_compression:
            return context
        
        current_tokens = count_tokens(context)
        available_tokens = self.config.get_available_tokens()
        
        if current_tokens <= available_tokens:
            return context
        
        # simple truncation strategy (keep first N tokens)
        print(f"⚠️ Context exceeds budget ({current_tokens} > {available_tokens}), performing truncation")
        
        # truncate based on paragraph to preserve structure
        lines = context.split("\n")
        compressed_lines = []
        used_tokens = 0
        
        for line in lines:
            line_tokens = count_tokens(line)
            if used_tokens + line_tokens > available_tokens:
                break
            compressed_lines.append(line)
            used_tokens += line_tokens
        
        return "\n".join(compressed_lines)


# Module-level cached tiktoken encoding (created once on first use)
_TIKTOKEN_ENC = None


def _get_tiktoken_enc():
    """Return the cached tiktoken encoding, creating it on first call."""
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is None:
        try:
            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            pass
    return _TIKTOKEN_ENC


def count_tokens(text: str) -> int:
    """count token in text using tiktoken (cached encoder)"""
    enc = _get_tiktoken_enc()
    if enc is not None:
        return len(enc.encode(text))
    return len(text) // 4

