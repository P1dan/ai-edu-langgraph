from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings

def build_vector_db():
    loader = TextLoader(
        "../../data/rag_data/三角函数学习路径.md",
        encoding="utf-8"
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = splitter.split_documents(docs)

    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key="sk-30b0ba857316437087ace218df67aa95"
    )

    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local("vector_db/trigonometry")

    print("✅ 向量库构建完成（阿里 Embedding）")

if __name__ == "__main__":
    build_vector_db()
