from typing import List, Mapping


depth_base_instruction = """I want you act as a Prompt Rewriter.\n 
Your objective is to rewrite a {Given Prompt} into a **more complex version** to make those famous AI systems (e.g., ChatGPT) a bit harder to handle.\n 
**Requirements**:\n
- The {Rewritten Prompt} cannot omit the input in the {Given Prompt}. \n 
- You SHOULD complicate the given prompt using the following method: \n
<STRATEGY> \n
- The {Rewritten Prompt} must be reasonable and must be understood and responded by humans.\n
**Constraints that must be followed**:\n
- The {Rewritten Prompt} can only add 10 to 20 words into the {Given Prompt}. \n
- The {Rewritten Prompt} should be self-contained, with **all necessary information** provided, so that it can be responded to without needing to refer back to the the {Given Prompt}.\n
- Your response should contain **only** the {Rewritten Prompt}, **without any** additional formatting or introductory phrases such as 'Here is the rewritten prompt:' or 'The rewritten prompt is:'.\n
"""


def createConstraintsPrompt(instruction):
	prompt = depth_base_instruction.replace("<STRATEGY>", "Please add one more constraints/requirements into the {Given Prompt}")
	prompt += "The {Given Prompt}: \n<PROMPT>\n".replace('<PROMPT>', instruction)
	prompt += "========\nBased on the prompt above, rewrite a prompt:\n{Rewritten Prompt}:\n"
	return prompt

def createDeepenPrompt(instruction):
	prompt = depth_base_instruction.replace("<STRATEGY>", "If the {Given Prompt} contains inquiries about certain issues, the depth and breadth of the inquiry can be increased.")
	prompt += "The {Given Prompt}: \n<PROMPT>\n".replace('<PROMPT>', instruction)
	prompt += "========\nBased on the prompt above, rewrite a prompt:\n{Rewritten Prompt}:\n"
	return prompt

def createConcretizingPrompt(instruction):
	prompt = depth_base_instruction.replace("<STRATEGY>", "Please replace general concepts with more specific concepts.")
	prompt += "The {Given Prompt}: \n<PROMPT>\n".replace('<PROMPT>', instruction)
	prompt += "========\nBased on the prompt above, rewrite a prompt:\n{Rewritten Prompt}:\n"
	return prompt


def createReasoningPrompt(instruction):
	prompt = depth_base_instruction.replace("<STRATEGY>", "If the {Given Prompt} can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.")
	prompt += "The {Given Prompt}: \n<PROMPT>\n".replace('<PROMPT>', instruction)
	prompt += "========\nBased on the prompt above, rewrite a prompt:\n{Rewritten Prompt}:\n"
	return prompt

breadth_base_instruction = """
I want you to act as a Prompt Creator.\n
Your objective is to take inspiration from the {Given Prompt} to create **one** brand new prompt.\n
**Reqiuirements**:\n
- This new {Created Prompt} should belong to the same domain as the {Given Prompt} but with different details.\n
- The LENGTH and complexity of the {Created Prompt} should be similar to that of the {Given Prompt}.\n
- The {Created Prompt} must be reasonable and must be understood and responded by humans.\n
- If the {Given Prompt} includes a specific input as part of its instructions, create a new input for your {Created Prompt} when applicable.\n
**Constraints that must be followed**:\n
- The {Created Prompt} should be self-contained, with **all necessary information** provided, so that it can be responded to without needing to refer back to the {Given Prompt}.\n
- Your response should contain **only** the {Created Prompt}, **without any** additional formatting or introductory phrases such as 'Here is the created prompt:' or 'The created prompt is:'.\n
"""

def createBreadthPrompt(instruction):
	prompt = breadth_base_instruction
	prompt += "The {Given Prompt}: \n<PROMPT>\n".replace('<PROMPT>', instruction)
	prompt += "========\nBased on the prompt above, create your prompt:\n{Created Prompt}:\n"
	return prompt

evolve_strategies: List[Mapping[str, str]] = [createBreadthPrompt, createDeepenPrompt, createConstraintsPrompt, createConcretizingPrompt, createReasoningPrompt,]


if __name__ == '__main__':
	for s in evolve_strategies:
		print(s("PROMPT HERE"))