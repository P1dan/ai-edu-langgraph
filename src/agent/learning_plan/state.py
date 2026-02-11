from typing import Optional, List
from langgraph.graph import MessagesState


class LearningState(MessagesState):

    # ========= 用户输入 =========
    learning_goal: str = ""
    background: str = ""
    time_budget: str = ""

    # ========= 流程控制 =========
    current_stage: str = "init"
    review_round: int = 0
    max_review_round: int = 3

    # ========= 分析阶段 =========
    refined_goal: Optional[str] = None
    knowledge_context: Optional[str] = None
    learning_strategy: Optional[str] = None

    # ========= 计划版本管理 =========
    learning_plan_versions: List[str] = []
    learning_plan_final: Optional[str] = None

    # ========= 审核阶段 =========
    teacher_feedback: Optional[str] = None
    is_approved: Optional[bool] = None
