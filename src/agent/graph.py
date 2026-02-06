import os
from typing import TypedDict, Optional, cast
from langchain_community.embeddings import DashScopeEmbeddings
from langgraph.graph import StateGraph
from src.agent.AliyunLLM import AliyunLLMWrapper
from src.agent.rag_retriever import RAGRetriever

import os
from pathlib import Path
from langchain_community.embeddings import DashScopeEmbeddings
from src.agent.rag_retriever import RAGRetriever

# 设置阿里 API Key
os.environ["ALIYUN_API_KEY"] = "sk-30b0ba857316437087ace218df67aa95"

# graph.py 所在目录
AGENT_DIR = Path(__file__).resolve().parent

# 项目根目录
PROJECT_ROOT = AGENT_DIR.parent  # src/

# 向量数据库绝对路径
VECTOR_DB_PATH = PROJECT_ROOT / "vector_db" / "trigonometry"

# 1. State 定义
class LearningState(TypedDict):
    # -------- 用户输入 --------
    learning_goal: str
    background: str
    time_budget: str

    # -------- 中间状态 --------
    refined_goal: Optional[str]
    knowledge_context: Optional[str]
    learning_strategy: Optional[str]

    # -------- 最终输出 --------
    learning_plan_document: Optional[str]
# 2. LLM 初始化（ChatGPT）
# llm = ChatOpenAI(
#     model="gpt-3.5-turbo",
#     temperature=0.2,
# )
# 初始化 DeepSeek LLM
# llm = DeepSeekLLM(
#     api_key="你的_DEEPSEEK_API_KEY",
#     model="deepseek-v1",  # 具体模型名根据 DeepSeek 提供的版本
#     temperature=0.2,      # 控制生成随机性
# )
llm = AliyunLLMWrapper(
    model_name="qwen-plus",   #
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.7
)

# 初始化 RAG
rag = RAGRetriever(
    vector_db_path=str(VECTOR_DB_PATH),  # 转为字符串
    top_k=4,
)
# 3. 各节点定义
def refine_goal(state: LearningState) -> LearningState:
    """澄清学习目标"""
    prompt = f"""
你是学习规划专家。

【用户原始目标】
{state["learning_goal"]}

【用户背景】
{state["background"]}

【时间条件】
{state["time_budget"]}

请将该目标澄清为：
- 明确的学习终点
- 可执行、可评估
- 避免空泛表述

直接输出澄清后的目标。
"""
    res = llm.invoke(prompt)
    state["refined_goal"] = res.content
    return state

def retrieve_knowledge(state: LearningState) -> LearningState:
    """
    RAG：从向量数据库中检索学习经验 / 结构知识
    """
    query = f"""
学习目标：{state["refined_goal"]}
用户背景：{state["background"]}
请提供适合该目标的学习结构、阶段划分、注意事项
"""

    context = rag.retrieve(query)

    state["knowledge_context"] = context
    return state

def decide_strategy(state: LearningState) -> LearningState:
    """生成整体学习策略"""
    prompt = f"""
你是学习策略设计专家。

【澄清后的学习目标】
{state["refined_goal"]}

【用户背景】
{state["background"]}

【时间条件】
{state["time_budget"]}

【参考知识】
{state["knowledge_context"]}

请给出整体学习策略说明：
- 学习节奏
- 理论 vs 实践占比
- 阶段性重点

不需要列具体知识点。
"""
    res = llm.invoke(prompt)
    state["learning_strategy"] = res.content
    return state

def generate_learning_plan_document(state: LearningState) -> LearningState:
    """生成最终学习路径文档（融合：大纲 + 路线 + 时间）"""
    prompt = f"""
你需要生成一份【完整学习路径文档（Markdown）】。

【学习目标】
{state["refined_goal"]}

【用户背景】      
{state["background"]}

【整体学习策略】
{state["learning_strategy"]}

【总时间约束】
{state["time_budget"]}

【输出要求（非常重要）】：
1. 使用 Markdown
2. 按【阶段】组织（如：阶段一 / 阶段二）
3. 每个阶段包含多个【学习模块】
4. 每个学习模块必须包含：
   - 学习主题
   - 学习目标
   - 学习内容（具体知识或技能）
   - 前置依赖
   - 建议学习时间（明确到周或小时）
5. 时间分配要与总时间约束大致匹配
6. 只输出最终文档，不要解释

开始输出。
"""
    res = llm.invoke(prompt)
    state["learning_plan_document"] = res.content
    return state

# 4. LangGraph 组装
builder = StateGraph(LearningState)
builder.add_node("refine_goal", refine_goal)
builder.add_node("retrieve_knowledge", retrieve_knowledge)
builder.add_node("decide_strategy", decide_strategy)
builder.add_node("generate_learning_plan_document", generate_learning_plan_document)
builder.set_entry_point("refine_goal")
builder.add_edge("refine_goal", "retrieve_knowledge")
builder.add_edge("retrieve_knowledge", "decide_strategy")
builder.add_edge("decide_strategy", "generate_learning_plan_document")
graph = builder.compile()

# 5. 调用示例
if __name__ == "__main__":
    result = graph.invoke(
        cast(LearningState, {
            "learning_goal": "系统学习高中数学中的三角函数",
            "background": "我已经掌握初中数学的基础知识，包括代数、几何、初步概率和统计",
            "time_budget": "3 个月，每周 15 小时",
        })
    )
    # 7️⃣ 输出到 Markdown 文件
    output_file = "../rag_data/三角函数学习路径.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result["learning_plan_document"])
    print(f"\n学习路径文档已保存到 {output_file}")

