from langgraph.constants import END
from langgraph.graph import StateGraph
from agent.learning_plan.state import LearningState
from agent.learning_plan.nodes import (
    refine_goal,
    retrieve_knowledge,
    decide_strategy,
    generate_learning_plan_document,
    teacher_review,
    revise_plan,
)

MAX_REVIEW_ROUNDS = 3  # 防止无限循环

def build_learning_plan_graph():
    builder = StateGraph(LearningState)

    # ===== 基础节点 =====
    builder.add_node("refine_goal", refine_goal)
    builder.add_node("retrieve_knowledge", retrieve_knowledge)
    builder.add_node("decide_strategy", decide_strategy)
    builder.add_node("generate_learning_plan_document", generate_learning_plan_document)

    # ===== 审核闭环节点 =====
    builder.add_node("teacher_review", teacher_review)
    builder.add_node("revise_plan", revise_plan)

    # ===== 主流程 =====
    builder.set_entry_point("refine_goal")
    builder.add_edge("refine_goal", "retrieve_knowledge")
    builder.add_edge("retrieve_knowledge", "decide_strategy")
    builder.add_edge("decide_strategy", "generate_learning_plan_document")
    builder.add_edge("generate_learning_plan_document", "teacher_review")

    # ===== 条件分支（带轮次限制）=====
    def review_router(state: LearningState):

        # 如果还没审核，理论上不该进入这里
        if "is_approved" not in state:
            return "revise"

        if state["is_approved"]:
            return "approved"

        review_round = state.get("review_round", 0)

        if review_round >= MAX_REVIEW_ROUNDS:
            return "approved"

        return "revise"

    builder.add_conditional_edges(
        "teacher_review",
        review_router,
        {
            "approved": END,
            "revise": "revise_plan"
        }
    )

    # 修改后重新进入审核
    builder.add_edge("revise_plan", "teacher_review")

    return builder.compile()
