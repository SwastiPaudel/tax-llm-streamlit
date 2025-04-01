import os
from collections import namedtuple

from deepeval.dataset import EvaluationDataset, Golden
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, GEval
import dotenv
from typing import List
from deepeval.dataset import EvaluationDataset
from rag_methods import initialize_app, stream_llm_rag_response

openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

dotenv.load_dotenv(override=True)

dataset = EvaluationDataset()
dataset.pull(alias="tax-p17-golds", auto_convert_goldens_to_test_cases=False)


# gemini_llm = ChatGoogleGenerativeAI(
#     api_key=gemini_api_key,
#     model="gemini-2.0-flash",
#     temperature=0.2,
#     streaming=True
# )
#
anthropic_llm = ChatAnthropic(
    api_key=anthropic_api_key,
    model="claude-3-5-sonnet-20240620",
    temperature=0.2,
    streaming=True,
)

openai_llm = ChatOpenAI(
    api_key=openai_api_key,
    model_name="gpt-4o",
    temperature=0.2,
    streaming=True,
)

#
# def test_relevancy(input, actual_output, retrieval_context, threshold=0.5, model="gpt-4o"):
#     relevancy_metric = AnswerRelevancyMetric(threshold=threshold, model=model)
#     test_case_1 = LLMTestCase(
#         input=input,
#         actual_output=actual_output,
#         retrieval_context=retrieval_context,
#     )
#     assert_test(test_case_1, [relevancy_metric])

#
correctness_metric = GEval(
        name="Correctness",
        model="gpt-4o",
        evaluation_params=[
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT],
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
            "You should also lightly penalize omission of detail, and focus on the main idea",
            "Vague language, or contradicting OPINIONS, are OK"
        ],
    )
#
#
# first_test_case = LLMTestCase(input="What is an ITIN, who needs to apply for one, and what documentation is required?",
#                               actual_output="""An Individual Taxpayer Identification Number (ITIN) is a tax processing number issued by the Internal Revenue Service (IRS) for individuals who are required to have a U.S. taxpayer identification number but are not eligible to obtain a Social Security Number (SSN). This includes non-resident aliens, resident aliens (based on days present in the U.S.), and dependents or spouses of U.S. citizens/resident aliens who do not qualify for an SSN.
#
# To apply for an ITIN, individuals must complete Form W-7, "Application for IRS Individual Taxpayer Identification Number." Along with the form, applicants must provide documentation that proves their identity and foreign status. This documentation can include a passport, national identification card, U.S. driver's license, or a birth certificate, among others. The documentation must be submitted in original form or as certified copies from the issuing agency.""",
#                               expected_output="""An ITIN (Individual Taxpayer Identification Number) is a nine-digit number issued by the IRS for individuals who need a taxpayer identification number for federal tax purposes but are not eligible for a Social Security Number (SSN).
#
# Who needs to apply for one? Individuals who must furnish a taxpayer identification number include:
# 1. Nonresident aliens filing a U.S. federal tax return.
# 2. Individuals claiming tax treaty benefits.
# 3. Nonresident alien spouses of U.S. citizens filing jointly.
# 4. U.S. resident aliens not eligible for an SSN.
# 5. Certain dependents.
#
# Documentation required includes:
# - A completed Form W-7.
# - A valid federal income tax return unless an exception applies.
# - Supporting documents confirming identity and foreign status (e.g., passport, national identification card) must be submitted.
# If applying based on an SSN denial, attach the denial letter from the SSA.""")
#
#
initialize_app()


def convert_goldens_to_test_cases(goldens: List[Golden]) -> List[LLMTestCase]:
    test_cases = []
    for golden in goldens:
        questions = golden.input
        Message = namedtuple('Message', ['role', 'content'])
        messages = [Message(role="user", content=questions)]

        rag_output, rag_context, sources = stream_llm_rag_response(openai_llm, messages)

        test_case = LLMTestCase(
            input=golden.input,
            # Generate actual output using the 'input' and 'additional_metadata'
            actual_output=rag_output,
            expected_output=golden.expected_output,
            context=golden.context,
            retrieval_context=rag_context
        )
        test_cases.append(test_case)
    return test_cases


dataset.test_cases = convert_goldens_to_test_cases(dataset.goldens)
dataset.push(alias="tax-p17-openai", overwrite=False, auto_convert_test_cases_to_goldens=True)

# test_cases = [first_test_case]
#
#
evaluation_output = dataset.evaluate([correctness_metric])

