# Evaluation

We provide the configs used for evaluation using the [`lighteval`](https://github.com/huggingface/lighteval) library.

Danish translation literals were not available in `lighteval` at the time of this work and we used the following:
```python
TranslationLiterals(
    language=Language.DANISH,
    question_word="spørgsmål",
    answer="svar",
    confirmation_word="ikke sandt",
    yes="ja",
    no="nej",
    also="også",
    cause_word="fordi",
    effect_word="derfor",
    or_word="eller",
    true="sand",
    false="falsk",
    neither="ingen af delene",
    full_stop=".",
    comma=",",
    question_mark="?",
    exclamation_mark="!",
    word_space=" ",
    sentence_space=" ",
    colon=":",
    semicolon=";",
)
```