from datasets import load_dataset

import verifiers as vf


def load_environment(
    dataset_name: str = "spellCheckRL/spellCheckRL",
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

        def weighted_lev(x: str, y: str, len_x: float, count: int) -> float:
            """
            Return the weigthed Levenshtein distance between x and y.
            """
        
        response = parser.parse_answer(completion) or ""
        return weighted_lev(answer, response)

    rubric = vf.Rubric(
        funcs=[
            lcs_reward_func,
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
