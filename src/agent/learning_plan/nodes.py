import asyncio
from langgraph.types import interrupt
from agent.llm.AliyunLLM import AliyunLLMWrapper
from agent.rag.rag_retriever import RAGRetriever
from agent.config import VECTOR_DB_DIR
from .state import LearningState

# ===== 全局能力（只初始化一次）=====
llm = AliyunLLMWrapper(
    model_name="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.7,
)
rag = RAGRetriever(
    vector_db_path=str(VECTOR_DB_DIR),
    top_k=4,
)

async def refine_goal(state: LearningState) -> LearningState:
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
    state["refined_goal"] = await llm.invoke(prompt).content
    return state

async def retrieve_knowledge(state: LearningState) -> LearningState:
    query = f"""
学习目标：{state["refined_goal"]}
用户背景：{state["background"]}
请提供适合该目标的学习结构、阶段划分、注意事项
"""

    # IO操作放入线程池
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: rag.retrieve(query)
    )

    state["knowledge_context"] = result
    return state

async def decide_strategy(state: LearningState) -> LearningState:
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
"""
    state["learning_strategy"] = await llm.invoke(prompt).content
    return state

async def generate_learning_plan_document(state: LearningState):

    state["plan_status"] = "generating"

    prompt = f"""
    你需要生成一份【完整学习路径文档（Markdown）】。
    ...
    """

    response = await llm.ainvoke(prompt)
    document = response.content

    state["learning_plan_document"] = document
    state["plan_status"] = "waiting_teacher_review"

    return interrupt(
        {
            "type": "plan_generated",
            "plan_id": state["plan_id"],
            "document": document
        }
    )

async def teacher_review(state: LearningState):

    # 第一次进入，等待老师输入
    if state.get("is_approved") is None:

        state["plan_status"] = "waiting_teacher_review"

        return interrupt(
            {
                "type": "teacher_review",
                "plan_id": state["plan_id"],
                "document": state["learning_plan_document"]
            }
        )

    # 恢复后执行
    if state["is_approved"]:

        state["review_status"] = "approved"
        state["plan_status"] = "approved"
        state["current_stage"] = "completed"

    else:

        state["review_status"] = "revise"
        state["plan_status"] = "revising"
        state["review_round"] = state.get("review_round", 0) + 1

        # 为下一轮清空审批状态
        state["is_approved"] = None

    return state

async def revise_plan(state: LearningState):

    # 根据 teacher_feedback 修改文档
    prompt = f"""
    老师反馈：
    {state["teacher_feedback"]}

    原始文档：
    {state["learning_plan_document"]}

    请根据反馈修改文档。
    """

    response = await llm.ainvoke(prompt)

    state["learning_plan_document"] = response.content
    state["plan_status"] = "waiting_teacher_review"

    # 重置审核状态
    state["is_approved"] = None
    state["teacher_feedback"] = None

    return state



