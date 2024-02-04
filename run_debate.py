import argparse
import json
import logging
import os
import pandas as pd
import pprint
import yaml

from common import RandomnessSource
from debate import Debate
from logger_utils import maybe_log
from tqdm import tqdm

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
DEBATER_SYS_PROMPT_FP = os.path.join(__location__, "prompts/debater_sys_prompt.txt")
JUDGE_SYS_PROMPT_FP = os.path.join(__location__, "prompts/judge_sys_prompt.txt")


def load_quality_data(
    data_dir, split="train", sample_size=None, seed=None, subset="difficult"
):
    fp = os.path.join(data_dir, f"QuALITY.v1.0.1.htmlstripped.{split}")
    data = [json.loads(line) for line in open(fp).readlines()]
    # filter the questions for each example, also split data into passages and questions
    passages_list = []
    passages_cols = [
        "article_id",
        "source",
        "title",
        "author",
        "topic",
        "url",
        "year",
        "license",
        "article",
    ]
    questions_list = []
    questions_parent_cols = [
        "set_unique_id",
        "writer_id",
        "article_id",
    ]
    questions_cols = [
        "question",
        "options",
        "best_distractor",
        "gold_label",
        "writer_label",
        "difficult",
    ]
    for ex in data:
        passages_list.append({col: ex[col] for col in passages_cols})
        if subset == "difficult":
            qs = [
                q
                for q in ex["questions"]
                if q["difficult"] == 1
                and all([vv["untimed_eval2_context"] > 1 for vv in q["validation"]])
            ]
        else:
            qs = [q for q in ex["questions"] if q["difficult"] != 1]
        ex["questions"] = qs
        # choose best distractor by majority vote
        for q in ex["questions"]:
            best_distractor_votes = [
                vv["untimed_best_distractor"] for vv in q["validation"]
            ]
            q["best_distractor"] = max(
                set(best_distractor_votes), key=best_distractor_votes.count
            )
            q_dict = {col: ex[col] for col in questions_parent_cols}
            q_dict = {**q_dict, **{col: q[col] for col in questions_cols}}
            questions_list.append(q_dict)
    passages_df = pd.DataFrame(passages_list).drop_duplicates(subset=["article_id"])
    questions_df = pd.DataFrame(questions_list)
    # now assign question IDs
    questions_df["question_id"] = questions_df.groupby(
        ["article_id", "question"]
    ).ngroup()
    for _, row in questions_df.iterrows():
        if row["gold_label"] == row["best_distractor"]:
            logging.warning(
                f"Gold label is the same as the best distractor for article {row['article_id']}, question {row['question_id']}. This question will be removed."
            )
    questions_df = questions_df[
        questions_df["gold_label"] != questions_df["best_distractor"]
    ]

    if sample_size is not None:
        questions_df = questions_df.sample(n=sample_size, random_state=seed)
    # Keep only the passages for which we have questions
    question_article_ids = set(questions_df["article_id"])
    passages_df = passages_df[passages_df["article_id"].isin(question_article_ids)]

    return passages_df, questions_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", default=None, type=int)
    parser.add_argument(
        "--data-dir",
        type=str,
        default="quality_dataset",
        help="Where the QuALITY dataset is located.",
    )
    parser.add_argument(
        "--difficulty-level",
        type=str,
        default="difficult",
        choices=["easy", "difficult"],
        help="Whether to use the difficult or non-difficult subset of QuALITY.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--output-filename-prefix",
        type=str,
        default="debate",
        help="Will output three files - {prefix}_debate_transcripts.jsonl, {prefix}_questions.jsonl, {prefix}_passages.jsonl",
    )
    parser.add_argument(
        "--loglevel",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
    )
    parser.add_argument(
        "--arch",
        default="gpt-4-1106-preview",
        choices=[
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-4",
            "gpt-4-1106-preview",
        ],
        help="Which OpenAI model to use as the debaters and judge.",
    )
    parser.add_argument(
        "--debater-temperature",
        default=0.8,
        type=float,
        help="If a debater is using temperature sampling, what temperature will be used.",
    )
    parser.add_argument(
        "--judge-temperature",
        default=0.0,
        type=float,
        help="What temperature the judge samples with.",
    )

    # Debate options
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--char-limit",
        type=int,
        default=500,
        help="Total character limit for arguments, including quotes.",
    )
    parser.add_argument(
        "--quote-char-limit",
        type=int,
        default=500,
        help="Character limit for quotes.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="The maximum number of rounds of debate.",
    )
    parser.add_argument(
        "--num-few-shot-examples",
        type=int,
        default=2,
        help="Num. of few-shot examples to use.",
    )
    parser.add_argument(
        "--max-debater-length-retries",
        type=int,
        default=2,
        help="The maximum num. of times that the debater can try to get their response under the character limit. Must be >=1.",
    )
    parser.add_argument(
        "--max-judge-format-retries",
        type=int,
        default=3,
        help="The maximum num. of times that the judge can try to reformat their response correctly. Must be >=1.",
    )
    parser.add_argument(
        "--output-transcripts",
        action="store_true",
        help="Whether to output each debate transcript.",
    )
    parser.add_argument(
        "--transcripts-output-dir",
        type=str,
        default="debate_transcripts",
        help="Directory to output debate transcripts in.",
    )
    parser.add_argument(
        "--randomness-source",
        choices=RandomnessSource._member_names_,
        default=[],
        action="append",
    )
    parser.add_argument(
        "--debater-to-assign-random-intervention-to",
        type=str,
        choices=["correct", "incorrect", "none"],
        help="If adding randomness, which debater to assign the random treatment to.",
        default="none",
    )
    parser.add_argument(
        "--randomness-mean",
        type=float,
        default=0.0,
        help="Mean of Gaussian noise added to judge probabilities when either JUDGE_PROB or JUDGE_PROB_SINGLE_DEBATER are used as randomness sources.",
    )
    parser.add_argument(
        "--randomness-std",
        type=float,
        default=0.1,
        help="Std. dev. of Gaussian noise added to judge probabilities when either JUDGE_PROB or JUDGE_PROB_SINGLE_DEBATER are used as randomness sources.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Load options from a config file instead. Overrides command-line arguments.",
    )

    args = parser.parse_args()
    if args.config is not None:
        config_args = yaml.safe_load(open(args.config))
        for k, v in config_args.items():
            if hasattr(args, k):
                setattr(args, k, v)
            else:
                logging.warning(
                    f"Config file {args.config} contains property {k} but this is not a valid command-line argument. Skipping."
                )
        # args = argparse.Namespace(**yaml.load(open(args.config).read()))
    logging.basicConfig(level=args.loglevel.upper())
    argsdict = vars(args)
    logging.info(f"Running {__file__} with the following argments:")
    logging.info(pprint.pformat(argsdict))
    return args


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.transcripts_output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, f"{args.output_filename_prefix}_args.jsonl"), "w"
    ) as f:
        f.write(pprint.pformat(vars(args)))
    passages_df, questions_df = load_quality_data(
        args.data_dir,
        split="train",
        sample_size=args.sample_size,
        seed=args.seed,
        subset=args.difficulty_level,
    )
    passages_df = passages_df.set_index("article_id")
    passages_df.to_json(
        os.path.join(args.output_dir, f"{args.output_filename_prefix}_passages.jsonl"),
        orient="records",
        lines=True,
    )
    questions_df.to_json(
        os.path.join(args.output_dir, f"{args.output_filename_prefix}_questions.jsonl"),
        orient="records",
        lines=True,
    )

    logger = logging.getLogger(name="debate")
    debates = []
    output_debates_fp = os.path.join(
        args.output_dir, f"{args.output_filename_prefix}_debates.jsonl"
    )
    # For re-starting partially completed experiments
    already_conducted_debates = {}
    num_prev_debates = 0
    if os.path.exists(output_debates_fp):
        maybe_log(
            logger,
            f"{output_debates_fp} already exists. Loading previous results...",
            level="info",
        )
        prev_debates = pd.read_json(output_debates_fp, orient="records", lines=True)
        num_prev_debates = len(prev_debates)
        for _, row in prev_debates.iterrows():
            if row["article_id"] not in already_conducted_debates:
                already_conducted_debates[row["article_id"]] = {row["question_id"]}
            else:
                already_conducted_debates[row["article_id"]].add(row["question_id"])
    if num_prev_debates <= args.sample_size:
        questions_df = questions_df.iloc[: (args.sample_size - num_prev_debates)]
        maybe_log(logger, f"Continuing with {len(questions_df)} debates.")
    else:
        maybe_log(
            logger,
            f"Previous results from {output_debates_fp} already contain more results than --sample-size={args.sample_size}. Exiting.",
        )
        return
    for _, row in tqdm(questions_df.iterrows()):
        if (
            row["article_id"] in already_conducted_debates
            and row["question_id"] in already_conducted_debates[row["article_id"]]
        ):
            maybe_log(
                logger,
                f"Debate for article {row['article_id']}, question {row['question_id']} already exists. Skipping.",
                level="info",
            )
            continue
        maybe_log(
            logger,
            f"Running debate for article {row['article_id']}, question {row['question_id']}.",
            level="info",
        )

        story = passages_df.loc[row["article_id"]]["article"]
        question = row["question"]
        correct_answer = row["options"][row["gold_label"] - 1]
        best_distractor = row["options"][row["best_distractor"] - 1]
        randomness_sources = [RandomnessSource[rs] for rs in args.randomness_source]
        deb_obj = Debate(
            story,
            question,
            correct_answer,
            best_distractor,
            args.char_limit,
            args.quote_char_limit,
            model=args.arch,
            max_rounds=args.max_rounds,
            logger=logger,
            debater_temperature=args.debater_temperature,
            judge_temperature=args.judge_temperature,
            num_few_shot_examples=args.num_few_shot_examples,
            max_debater_length_retries=args.max_debater_length_retries,
            max_judge_format_retries=args.max_judge_format_retries,
            randomness_sources=randomness_sources,
            debater_to_assign_random_intervention_to=args.debater_to_assign_random_intervention_to,
            mean=args.randomness_mean,
            std=args.randomness_std,
        )
        history = deb_obj.run_debate()
        debates.append(
            {
                "article_id": row["article_id"],
                "set_unique_id": row["set_unique_id"],
                "question_id": row["question_id"],
                "question": row["question"],
                "correct_answer_text": correct_answer,
                "best_distractor_text": best_distractor,
                "history": history,
                "answers": deb_obj.answers,
                "correct_debater": deb_obj.correct_debater_name,
                "distractor_debater": deb_obj.distractor_debater_name,
                "debater_a_temperature": deb_obj.debater_a.temperature,
                "debater_b_temperature": deb_obj.debater_b.temperature,
                "debater_a_adds_randomness_to_judge_probs": deb_obj.debater_a.debater_adds_randomness,
                "debater_b_adds_randomness_to_judge_probs": deb_obj.debater_b.debater_adds_randomness,
            }
        )
        if args.output_transcripts:
            output_fp = os.path.join(
                args.transcripts_output_dir,
                f"{args.output_filename_prefix}_article_{row['article_id']}_question_{row['question_id']}.txt",
            )
            transcript = deb_obj.prepare_transcript()
            with open(output_fp, "w") as f_out:
                f_out.write(transcript)
                maybe_log(
                    logger,
                    f"Wrote debate transcript for article {row['article_id']}, question {row['question_id']} to {output_fp}.",
                )
        with open(output_debates_fp, "a") as f_out:
            f_out.write(json.dumps(debates[-1]) + "\n")


if __name__ == "__main__":
    main(parse_args())
