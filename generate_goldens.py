from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.synthesizer import Synthesizer
from dotenv import load_dotenv

load_dotenv(override=True)

dataset = EvaluationDataset()

synthesizer = Synthesizer(model='gpt-4o-mini')  # Overwrite to use GPT4o for now

dataset.generate_goldens_from_docs(
    synthesizer=synthesizer,
    document_paths=["p17.pdf"],
    #context_construction_config=context_config,
)

dataset.save_as(file_type="csv", directory="./")
dataset.push('tax-p17-golds', overwrite=True, auto_convert_test_cases_to_goldens=True)



