from datasets import load_dataset

import verifiers as vf


def load_environment(
    dataset_name: str = "SykChw/obscureSpellCheck",
    dataset_split: str = "train",
    system_prompt: str
    | None = "Check the word character-by-character and suggest the closest correct spelling. Put your answer in <correct> tags.",
) -> vf.Environment:
    train_dataset = load_dataset(dataset_name, split=dataset_split).map(
        lambda x: {
            "question": x["prompt"],
            "answer": x["prompt"][::-1],
            "info": {},
            "task": "spell-check",
        }
    )
    train_dataset = train_dataset.remove_columns(["prompt"])

    parser = vf.XMLParser(["correct_spelling"], answer_field="correct_spelling")

    def lev_reward_func(completion, answer, **kwargs) -> float:
        """
        weigthed Levenshtein distance of the correctly spelled prompt and the parsed completion.
        """

        def weighted_lev(x: str, y: str, alpha: float = 0.7, beta: float = 0.3) -> float:
            """
            Compute a weighted Levenshtein reward between two strings x and y.
            - Substitution cost depends on ASCII distance between chars.
            - Position cost penalizes differences more at later positions.
            - Returns a reward in [0, 1], where 1 = exact match.
            """

            # Normalize case
            x, y = x.lower(), y.lower()
            m, n = len(x), len(y)

            # Handle empty strings
            if m == 0 and n == 0:
                return 1.0
            if m == 0 or n == 0:
                return 0.0

            # Initialize DP matrix
            dp = [[0.0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j

            # Fill DP
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        cost = 0.0
                    else:
                        # ASCII-based difference [0, 1]
                        char_diff = abs(ord(x[i - 1]) - ord(y[j - 1])) / 26.0
                        # Positional weighting [0, 1]
                        pos_weight = (i + j) / (m + n)
                        cost = alpha * char_diff + beta * pos_weight

                    dp[i][j] = min(
                        dp[i - 1][j] + 1,        # deletion
                        dp[i][j - 1] + 1,        # insertion
                        dp[i - 1][j - 1] + cost  # substitution
                    )

            # Normalize distance to [0, 1]
            distance = dp[m][n] / max(m, n)
            reward = 1.0 - min(distance, 1.0)

            return round(reward, 4)

    rubric = vf.Rubric(
        funcs=[
            lev_reward_func,
        ],
        weights=[1.0],
    )

    vf_env = vf.SingleTurnEnv(
        dataset=train_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
    return vf_env
