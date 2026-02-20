"""
Evolution strategy prompts for Evol-Instruct style math problem generation.

Each strategy: (system_prompt, user_template)
- user_template must contain {problem} placeholder
"""

STRATEGIES = {
    "harder": (
        "You are a math problem designer. Your task is to take an existing math problem "
        "and create a harder version. Increase the difficulty by adding more steps, "
        "requiring deeper reasoning, or introducing additional constraints. "
        "Output ONLY the new problem statement, nothing else.",
        "Make the following math problem significantly harder while keeping the same core concept:\n\n{problem}"
    ),

    "rewrite": (
        "You are a math problem designer. Your task is to rewrite a math problem "
        "into a completely different form while testing similar mathematical skills. "
        "Change the context, numbers, and wording. "
        "Ensure the rewritten problem is mathematically valid. "
        "Output ONLY the new problem statement, nothing else.",
        "Rewrite the following math problem into a different but equally challenging problem "
        "that tests similar skills:\n\n{problem}"
    ),

    "algebraize": (
        "You are a math problem designer. Your task is to generalize a concrete math problem "
        "by replacing specific numbers with variables, turning it into an algebraic problem. "
        "The result should require symbolic manipulation or finding a general formula. "
        "Output ONLY the new problem statement, nothing else.",
        "Convert the following problem into a more general algebraic form by replacing "
        "specific numbers with variables (e.g., n, k, a, b). "
        "The problem should ask for a general formula or expression:\n\n{problem}"
    ),

    "apply": (
        "You are a math problem designer. Your task is to transform an abstract math problem "
        "into a real-world application problem. Use contexts like physics, engineering, "
        "economics, biology, or everyday life. Keep the underlying math the same or harder. "
        "Output ONLY the new problem statement, nothing else.",
        "Transform the following math problem into a real-world application problem. "
        "Choose a realistic context (physics, finance, engineering, etc.) "
        "while preserving the core mathematical reasoning:\n\n{problem}"
    ),

    "compose": (
        "You are a math problem designer. Your task is to create a new problem that combines "
        "the mathematical concept in the given problem with another area of math "
        "(e.g., combine algebra with geometry, number theory with probability, "
        "calculus with combinatorics). The result should be a multi-step problem. "
        "Output ONLY the new problem statement, nothing else.",
        "Create a new multi-step problem that combines the concept from the following problem "
        "with a different area of mathematics:\n\n{problem}"
    ),

    "competition": (
        "You are a math olympiad problem designer. Your task is to transform a math problem "
        "into competition-style (AMC/AIME/Olympiad). The problem should require creative "
        "insight, have an elegant solution, and result in a clean numerical answer. "
        "Output ONLY the new problem statement, nothing else.",
        "Transform the following problem into a math competition style problem "
        "(AMC/AIME level). It should require creative insight and have a clean answer:\n\n{problem}"
    ),
}

# Weights for random strategy selection (favor harder + competition for quality)
STRATEGY_WEIGHTS = {
    "harder": 0.25,
    "rewrite": 0.15,
    "algebraize": 0.15,
    "apply": 0.15,
    "compose": 0.15,
    "competition": 0.15,
}

# Multi-round evolution: after first evolution, apply these follow-ups
FOLLOWUP_STRATEGIES = ["harder", "competition", "compose"]
