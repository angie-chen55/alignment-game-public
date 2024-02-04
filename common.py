from enum import Enum


class RandomnessSource(Enum):
    NONE = 1
    JUDGE_PROB = 2  # Both debaters are given noised judge probabilities
    DECODING_STRAT_SINGLE_DEBATER = 3  # One debater is assigned greedy decoding while the other uses temperature sampling
    RANDOM_STR = 4
    JUDGE_PROB_SINGLE_DEBATER = 5  # One debater is given noised judge probabilities
    DECODING_STRAT_BOTH = 6  # Both debaters use temperature sampling
