from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List

class RAGRetriever:
    """
    统一 RAG 检索模块
    - 只负责：向量检索
    - 不关心 LangGraph / Agent / Prompt
    """

    def __init__(
        self,
        vector_db_path: str,
        top_k: int = 4,
    ):
        self.vector_db_path = vector_db_path
        self.top_k = top_k

        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.load_local(
            vector_db_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def retrieve(self, query: str) -> str:
        """
        输入 query
        输出：拼接好的知识上下文（给 LLM 用）
        """
        docs: List[Document] = self.vectorstore.similarity_search(
            query, k=self.top_k
        )

        if not docs:
            return "（未检索到相关知识）"

        context = "\n\n".join(
            f"- {doc.page_content.strip()}" for doc in docs
        )

        return context
