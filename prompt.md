# 训练 LongCat-Flash-Chat 模型

1. 在 /home/jianzhnie/llmtuner/llm/torchtitan-npu 新增对 meituan-longcat/LongCat-Flash-Chat 模型的支持
2. 请帮我调试和训练 meituan-longcat/LongCat-Flash-Chat 模型
    - 模型路径： /home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Chat
    - 训练数据： /home/jianzhnie/llmtuner/hfhub/datasets/tatsu-lab/alpaca
    - docker 镜像：  torchtitan-npu:cann9.0.0-torch2.12.0


# 训练 Qwen3-30B-A3B 模型

1. 在 /home/jianzhnie/llmtuner/llm/torchtitan-npu 新增对 Qwen/Qwen3-30B-A3B 模型的支持
2. 请帮我调试和训练 Qwen/Qwen3-30B-A3B 模型
    - 模型路径： /home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-30B-A3B
    - 训练数据： /home/jianzhnie/llmtuner/hfhub/datasets/tatsu-lab/alpaca
    - docker 镜像：  torchtitan-npu:cann9.0.0-torch2.12.0
3. 通过多种方式优化 Qwen/Qwen3-30B-A3B 模型的性能, MFU， 等指标
4. 记得使用全量模型参数，不要使用部分参数