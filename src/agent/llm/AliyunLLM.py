import os
import requests


class AliyunLLMWrapper:
    """
    简单封装阿里云 LLM，兼容 .invoke(prompt)
    支持 OpenAI 兼容模式：
      - 模型示例: 'qwen-mini', 'qwen-plus'
      - URL 示例: https://dashscope.aliyuncs.com/compatible-mode/v1
    """

    def __init__(self, model_name="qwen-mini", temperature=0.7, base_url=None):
        self.api_key = os.environ.get("ALIYUN_API_KEY")
        if not self.api_key:
            raise ValueError("请在环境变量 ALIYUN_API_KEY 中配置你的 API Key")

        self.model_name = model_name
        self.temperature = temperature
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def invoke(self, prompt):
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }
        resp = requests.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"阿里云 LLM 调用失败: {resp.status_code} {resp.text}")

        data = resp.json()
        # OpenAI 兼容模式返回结构类似 chat/completions
        return type("Obj", (), {"content": data["choices"][0]["message"]["content"]})()
