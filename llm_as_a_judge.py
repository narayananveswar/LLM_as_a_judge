from unitxt import get_logger
from unitxt.api import create_dataset, evaluate
from unitxt.blocks import Task
from unitxt.inference import (
    HFPipelineBasedInferenceEngine,
)
from unitxt.llm_as_judge import LLMAsJudge
from unitxt.templates import InputOutputTemplate

# Initialize a logger for the Unitxt library. This helps in tracking the progress and any potential issues.
logger = get_logger()

# First, we define the examples data we want to evaluate using LLM as judge.
# Each dictionary in this list represents a single test case with a query, a relevant document, and the expected correct answer.
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
# Second, We define the prompt we show to the judge (the LLM).
# This template instructs the LLM to act as an impartial judge and evaluate the correctness of an assistant's answer.
# It specifies the format for the input (user's question and assistant's answer) and the desired output format for the verdict (either "[[10]]" for correct or "[[0]]" for wrong).
# It also includes a postprocessor to extract the numerical rating from the LLM's response.
judge_correctness_template = InputOutputTemplate(
    instruction="Please act as an impartial judge and evaluate if the assistant's answer is correct."
    ' Answer "[[10]]" if the answer is accurate, and "[[0]]" if the answer is wrong. '
    'Please use the exact format of the verdict as "[[rate]]". '
    "You can explain your answer after the verdict"
    ".\n\n",
    input_format="[User's input]\n{question}\n[Assistant's Answer]\n{answer}\n",
    output_format="[[{rating}]]",
    postprocessors=[
        r"processors.extract_mt_bench_rating_judgment", # This postprocessor is designed to extract the rating (10 or 0) from the LLM's output.
    ],
)

# Third, We define the metric as LLM as a judge, with the desired platform and model.
# This section configures the LLMAsJudge metric.
# inference_model specifies which LLM from Hugging Face Transformers will be used as the judge.
# template refers to the prompt template defined above that will be used to instruct the judge.
# task specifies the type of rating task.
# main_score indicates the name of the score that will be extracted from the judge's output.
# strip_system_prompt_and_format_from_inputs is set to False, meaning the input format will be preserved when passed to the judge.
llm_judge_metric = LLMAsJudge(
    inference_model=HFPipelineBasedInferenceEngine(
        model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct", max_new_tokens=32 # Specifies the Hugging Face model to use for judging and the maximum number of tokens for its response.
    ),
    template=judge_correctness_template,
    task="rating.single_turn",
    main_score="llm_judge_score",
    strip_system_prompt_and_format_from_inputs=False,
)
# This is the task card
# The Task object defines the inputs, references, prediction type, and the metrics to be used for evaluation.
task = Task(
    input_fields={"query": str, "document": str}, # Defines the input fields expected in the data.
    reference_fields={"reference_answer": str}, # Defines the reference (ground truth) field.
    prediction_type=str, # Specifies that the model will output a string as its prediction.
    metrics=[llm_judge_metric], # Assigns the LLMAsJudge metric to this task.
)

# This template is used to format the input that will be fed to the model being tested (SmolLM2 in this case).
# It takes the 'document' and 'query' and formats them into a natural language prompt.
template = InputOutputTemplate(
    instruction="Answer the following query based on the provided document.",
    input_format="Document:\n{document}\nQuery:\n{query}",
    output_format="{reference_answer}", # Specifies that the model should output the answer.
    postprocessors=["processors.lower_case"], # Converts the model's output to lowercase for potential case-insensitive comparison.
)

# Creates a dataset object using the defined data, task, and template.
# It specifies that the 'data' list should be used as the test set.
# max_test_instances limits the number of test instances to be used (here, it's set to 10, but the 'data' only has 2 examples).
dataset = create_dataset(
    test_set=data,
    task=task,
    template=template,
    split="test",
    max_test_instances=10,
)

# Infer using SmolLM2 using HF API
# This creates an inference engine based on the same SmolLM2 model.
# This engine will be used to generate predictions for the input data.
model = HFPipelineBasedInferenceEngine(
    model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct", max_new_tokens=32
)
# Generate predictions using the inference engine on the created dataset.
predictions = model(dataset)

# Evaluate the predictions using the defined metric.
# This function takes the generated predictions and the dataset (which contains the reference answers)
# and calculates the defined metrics (in this case, the LLMAsJudge metric).
results = evaluate(predictions=predictions, data=dataset)

# Print the global results of the evaluation.
# This will typically include the score calculated by the LLM judge.
print("Global Results:")
print(results.global_scores.summary)