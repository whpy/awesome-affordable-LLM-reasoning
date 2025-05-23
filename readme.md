# Awesome affordable methods for LLM reasoning in engineering

This is a collection of research papers & blogs for affordable methods capable of enhancing reasoning ability of LLM .

And the repository will be continuously updated to track the frontier of LLM Reasoning.

## Papers
* 
* 02 May 2025 [LoRI: Reducing Cross-Task Interference in Multi-Task Low-Rank Adaptation](https://arxiv.org/abs/2504.07448): In this work, the authors highlight that LoRA matrices still contain a considerable amount of redundant parameters. To address this, they propose LoRI, which significantly reduces the number of trainable parameters while maintaining strong task performance.
* 25 Apr 2025 [RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning](https://www.arxiv.org/abs/2504.20073): A comprehensive work about applying GRPO method on training agent to accomplish multi-round tasks.
* 15 Apr 2025 [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/html/2504.11536v1):This work enhances long-form reasoning with tool-integrated learning, including two key features: (1) dynamic interleaving of real-time code execution within natural language reasoning processes, and (2) an automated RL paradigm that allows policy rollouts with multi-turn real-time code execution and teaches the model in learning when and how to invoke tools based on outcome feedback. 
* 14 Apr 2025 [MegaMath: Pushing the Limits of Open Math Corpora](https://arxiv.org/abs/2504.02807): An open, large-scale, high-quality corpus tailored to the demands of math-centric LLM pre-training.
* 08 Apr 2025 [Deepcoder](https://www.together.ai/blog/deepcoder): A 14B reasoning model that finetuned for coding. Notably, they release a detail recipe for RL finetuning including dataset, environments and RL method used.
* [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118): An RL method for training reasoning model with higher efficiency. 
* 12 Mar 2025 [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516): This work presents a foundational framework and workflow for training large language models (LLMs) to handle retrieval-augmented generation (RAG), aiming to enhance reasoning performance using methods such as GRPO. The study yields several notable findings: (1) GRPO demonstrates faster convergence than PPO across all evaluated settings; (2) both PPO and GRPO achieve comparable final reward performance.
Compared with the 14B model, SEARCH-R1 consistently outperforms all baseline methods across the evaluated metrics. Furthermore, increasing the model size leads to consistent performance gains with SEARCH-R1, highlighting the benefits of LLM size scaling in our approach. 
The experiments are conducted using Qwen-2.5-3B and Qwen-2.5-7B as base models, trained on 8 × H100 GPUs. 

* 25 Feb 2025 [SoS1: O1 and R1-Like Reasoning LLMs are Sum-of-Square Solvers](https://arxiv.org/abs/2502.20545): For the model with size only 14B trained by 2 A100 on 1k dataset, SoS1 find out the new example of Hilbert problem.
* 11 Feb 2025 [CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction](https://arxiv.org/html/2502.07316v1#S3): In this work, DeepSeek introduce a new method to generate COT from the source code.
* 06 Feb 2025 [s1: Simple test-time scaling](https://arxiv.org/html/2501.19393v2): In this work, they claim that they fine-tune a small model (s1-32B, based on Qwen-series) that performs better than o1. There are two noted works. First, they curate a small dataset s1K of 1,000 questions paired with reasoning traces. Second, they develop budget forcing to control test-time compute by forcefully terminating the model’s thinking process. 
* 05 Feb 2024 [LIMO: Less is More for Reasoning](https://arxiv.org/pdf/2502.03387): They demonstrate a striking phenomenon: complex mathematical reasoning abilities can be effectively elicited with surprisingly few examples, merely 817 curated training samples.
* 22 Jan 2025 [DeepSeek-R1](https://arxiv.org/abs/2501.12948): the deepseek group release a powerful open-source model. They also release the small models distilled by the powerful model.
* 13 Jan 2025 [Sky-T1: Train your own O1 preview model within $450](https://novasky-ai.github.io/posts/sky-t1/): this repo release the data and the method about how to train a small model with outstanding reasoning ability as o1-preview. The claims that the model Sky-T1 is more well rounded over code, math, report, compared with qwq and o1.
* 06 Jan 2025 [Process Reinforcement Through Implicit Rewards](https://github.com/PRIME-RL/PRIME): This group release a new type of training which do not require explicit reward function. 
* 22 Dec 2024 [step\_noise: Training in 1.58b With No Gradient Memory](https://github.com/wbrickner/noise_step/tree/main): this work largely compress the model into a small size; furthermore, it proposes a method that doesn't require gradient.
* 02 Dec 2024 [MALT: Improving Reasoning with Multi-Agent LLM Training](https://arxiv.org/pdf/2412.01928)
It is about training multi-LLMagents system.
* 29 Nov 2024 [O1-CODER: AN O1 REPLICATION FOR CODING](https://arxiv.org/pdf/2412.00154)
* 18 Nov 2024 [ReST-MCTS∗: LLM Self-Training via Process Reward
Guided Tree Search](https://keg.cs.tsinghua.edu.cn/jietang/publications/NeurIPS24-Zhang-et-al-ReST-MCTS.pdf#:~:text=traces%20as%20well%20as%20per-step%20value%20to%20train,is%20able%20to%20infer%20the%20correct%20process%20reward)
This work use mcts to enhance the capability of reasoning. It is remarkable that in their github repo, there is only small amount of examples. It may indicate that such methods only require small amount of data.

* 05 Nov 2024 [Large Language Models Orchestrating Structured Reasoning Achieve Kaggle Grandmaster Leve](https://arxiv.org/pdf/2411.03562)
In this paper, LLM agent shows remarkable performance that achieving Kaggle Grandmaster in kaggle competition. It introduces a method to enhance the performance of LLM agent by utilizing the historical data.

* 12 Oct 2024 [OpenR: An Open Source Framework for Advanced Reasoning with Large Language Models](https://arxiv.org/abs/2410.09671)
This work release a light, integrated framework to facilitate the RL-based fine-tune of LLM.

* 04 Oct 2024 [Training Language Models to Self-Correct via Reinforcement Learning](https://arxiv.org/abs/2409.12917)

* 12 Aug 2024 [MUTUAL REASONING MAKES SMALLER LLMS STRONGER PROBLEM-SOLVERS](https://arxiv.org/pdf/2408.06195) [![](https://img.shields.io/badge/github-repo-blue)](https://github.com/zhentingqi/rStar)
This work investigates how to make small language models perform as well as the large models by introducing mcts-based methods.  

* 26 Jun 2024 [STEP-DPO: STEP-WISE PREFERENCE OPTIMIZATION FOR LONG-CHAIN REASONING OF LLMS](https://arxiv.org/abs/2406.18629)
By lots of experiments, this work draw a conclusion that training a LLM with step by step data could significantly enhance the ability of reasoning. They mention that step-DPO could even enhance the performance of LLM after RLHF, which indicates that step-DPO could complement RLHF effectively.

* 17 Jun 2024 [Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning](https://arxiv.org/pdf/2405.00451) [![](https://img.shields.io/badge/github-repo-blue)](https://github.com/YuxiXie/MCTS-DPO)

* 22 May 2024 [ReFT](https://arxiv.org/pdf/2404.03592)
They introduce a new type of fine-tunning based on reinforcement learning. It only requires a small amount of examples for fine-tunning a language model.

* 18 Mar 2024 [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)

## Blogs 
* [OneBit: Towards Extremely Low-bit Large Language Models](https://github.com/xuyuzhuang11/OneBit): A work to extremely quantize LLM.
* [Open-R1](https://huggingface.co/blog/open-r1): A project hosted by huggingface to reproduce DeepSeek-R1.
* [Required vram for LLMs fine-tunning in different size](https://www.reddit.com/r/LocalLLaMA/comments/18o5u0k/helpful_vram_requirement_table_for_qlora_lora_and/?rdt=48995)
* [Process Reinforcement through Implicit Rewards](https://curvy-check-498.notion.site/Process-Reinforcement-through-Implicit-Rewards-15f4fcb9c42180f1b498cc9b2eaf896f)
## Local configuration tricks
* [spectaculative decoding](https://arxiv.org/pdf/2401.07851v2)
* [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF): A lite library for RL-based methods in fine-tunning
## Powerful open-sourced small language models (SLM)
[DeepSeek_v3](https://huggingface.co/unsloth/DeepSeek-V3-GGUF): A Large model that could be run on local 4090Ti under Q2 quantization.

[BitNet](https://github.com/microsoft/BitNet): An architecture of LLM which manipulate several bits rather than those with long bytes.

[Qwen2.5-72B-instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct): Ranked 1st among open-sourced models on the coding leaderboard of [livebench](https://livebench.ai/#/?Reasoning=a&Coding=a&Mathematics=a&Data+Analysis=a).

[Qwen2.5-72B-instruct (quantized gguf)](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF/tree/main): quantized verison of Qwen2.5-72B

[Qwen2.5-Coder-32B](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct/tree/main): 32B sized model with outstanding performance on coding tasks, based on livebench score.

[QwQ-32B-preview](https://huggingface.co/Qwen/QwQ-32B-Preview): released by Ali. 

[marco-1](https://huggingface.co/AIDC-AI/Marco-o1): released by Ali. It utilizes the MCTS in reasoning. The performance on translation is emphasized in their paper.

[OpenCoder](https://huggingface.co/collections/infly/opencoder-672cec44bbb86c39910fb55e): claims that the performance on coding related task could 
### Benchmarks
[syslm](https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard): benchmarks claims that "hard for LLM to cheat."

[livebench](https://livebench.ai/#/?Reasoning=a&Coding=a&Mathematics=a&Data+Analysis=a): popular benchmark for evaluating performance of LLM.