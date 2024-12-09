we test the basic performance of different models in this folder

0. (test sample) Provide a Python code to simulate 2D cylinder flow using the Lattice Boltzmann Method (LBM). You could only use the numpy and matplotlib to complete this task. The flow moves left to right with an inlet velocity of 1.0, an equilibrium boundary condition at the outlet, and periodic boundaries on the top and bottom. The initial condition is zero velocity, the cylinder radius is 2.0, and the domain has a grid size of 100×50. The code should output the final flow field as a PNG image.
(prompt) -p "you are a helpful assistant in coding, specialized in python and LBM (CFD method). You would carefully output the program without bug. now take deep breath and do the assignments step by step.", we test the generated codes in teh perftest.py.

0.1 a less difficult task is to modify a given code files with known errors from compiler, the detail could be referred to opencoder.py.

0.2 now we are testing the models in the environment "conda activate huggingface"
0.3 the environment is recorded in environment.yml
0.4 always install the pytorch first for any environment

1. ./bin/llama-cli -m ./model/vicuna/vicuna-33b-coder.Q5_K_S.gguf -cnv -p "you are a good helper" -ngl 100, the parameter -ngl is essential to execute the models on gpu.

|name|perf|description|
|:---:|:---:|:---:|
|gemma-27B | 2-3 | could well follow the instructions. The conservation may collapse in two rounds. seems out of some weird reasons.|
| codellama | placeholder | |
|deepseekcoder-v2| placeholder ||
|qwen2.5-coder-32B-instruct| placeholder |It would cost very long time for inference. It performs the best in following the instructions. It shows a strong capability in long context follow.|
|qwen2.5-coder-72B-instruct-2bit| placeholder |For the same codes as others with errors from compiler, it still stucks at the error at line 44. We request it to generate step by step but that doesn't work. It may be due to the low precision of model we adopt, or it may be due to the intrinsic weakness at accurate counting. We may could try to introduce more information like adding the number of line in the source codes.|
|Opencoder8B|bad|cannot hold very long context (4096 tokens), it will refuse to give the complete code and bad at obeying the instruction. So it may be only fit for those simple code benchmark|
|marco-o1-7B|bad|cannot correct the bug by err-msg, and it is hard to trigger its mcts mode for reference.|
|QWQ-32B-preview|med--|for a simple code of lbm, given with the error from the compiler, it struggles to correct it. but at least each time the error is different. And compared with the gemma model, it could be interacted with for several turns. QWQ would give a very short reasoning chain, where it might be constrained by its size or the way Ali trains it?|
|llama-o1|bad|the model is bad at following the instructions. It refuse to give the complete corrected codes. And the response would start with strange serial numbers like "910", "911", "1213" (it may be the denotation of the mcts path?)|


for now, it seems that for the performance of transformer-based model, size matters a lot.
in this case, we could only focus on the test-time tricks.

for the large model (~70B), the time consumption for inference would signifcantly increase.

Maybe the depth of the model matters.