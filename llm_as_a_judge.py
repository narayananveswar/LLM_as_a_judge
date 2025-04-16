# As form the last linkdn post we had dicussed abot model test. Here is the code snippet for 
# using LLM for judging the model 
# note -- here I have used a model from hugging face so that it is easier to use it. using better models as judge improve the results 

from unitxt import get_logger
from unitxt.api import create_dataset, evaluate
from unitxt.blocks import Task
from unitxt.inference import (
    HFPipelineBasedInferenceEngine,
)
from unitxt.llm_as_judge import LLMAsJudge
from unitxt.templates import InputOutputTemplate

logger = get_logger()

# First, we define the examples data we want to evaluate using LLM as judge.
data = [
    {
        "query": "What is the capital of Texas?",
        "document": "The capital of Texas is Austin.",
        "reference_answer": "Austin",
    },
    {
        "query": "What is the color of the sky right now?",
        "document": "The sky is generally black during the night.",
        "reference_answer": "Black",
    },
]
# Second, We define the prompt we show to the judge.
judge_correctness_template = InputOutputTemplate(
    instruction="Please act as an impartial judge and evaluate if the assistant's answer is correct."
    ' Answer "[[10]]" if the answer is accurate, and "[[0]]" if the answer is wrong. '
    'Please use the exact format of the verdict as "[[rate]]". '
    "You can explain your answer after the verdict"
    ".\n\n",
    input_format="[User's input]\n{question}\n[Assistant's Answer]\n{answer}\n",
    output_format="[[{rating}]]",
    postprocessors=[
        r"processors.extract_mt_bench_rating_judgment",
    ],
)

# Third, We define the metric as LLM as a judge, with the desired platform and model.
llm_judge_metric = LLMAsJudge(
    inference_model=HFPipelineBasedInferenceEngine(
    model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct", max_new_tokens=32
    ),
    template=judge_correctness_template,
    task="rating.single_turn",
    main_score="llm_judge_score",
    strip_system_prompt_and_format_from_inputs=False,
)
# This is the task card 
task = Task(
    input_fields={"query": str, "document": str},
    reference_fields={"reference_answer": str},
    prediction_type=str,
    metrics=[llm_judge_metric],
)

template = InputOutputTemplate(
    instruction="Answer the following query based on the provided document.",
    input_format="Document:\n{document}\nQuery:\n{query}",
    output_format="{reference_answer}",
    postprocessors=["processors.lower_case"],
)

dataset = create_dataset(
    test_set=data,
    task=task,
    template=template,
    split="test",
    max_test_instances=10,
)

# Infer using SmolLM2 using HF API
model = HFPipelineBasedInferenceEngine(
    model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct", max_new_tokens=32
)
predictions = model(dataset)

# Evaluate the predictions using the defined metric.
results = evaluate(predictions=predictions, data=dataset)

print("Global Results:")
print(results.global_scores.summary)
