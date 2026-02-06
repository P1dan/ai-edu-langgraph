from langgraph.graph import StateGraph
from agent.learning_plan.state import LearningState
from agent.learning_plan.nodes import (
    refine_goal,
    retrieve_knowledge,
    decide_strategy,
    generate_learning_plan_document,
)


def build_learning_plan_graph():
    builder = StateGraph(LearningState)

    builder.add_node("refine_goal", refine_goal)
    builder.add_node("retrieve_knowledge", retrieve_knowledge)
    builder.add_node("decide_strategy", decide_strategy)
    builder.add_node("generate_learning_plan_document", generate_learning_plan_document)

    builder.set_entry_point("refine_goal")
    builder.add_edge("refine_goal", "retrieve_knowledge")
    builder.add_edge("retrieve_knowledge", "decide_strategy")
    builder.add_edge("decide_strategy", "generate_learning_plan_document")

    return builder.compile()
