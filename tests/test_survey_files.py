import os
from src.Surveyor import Surveyor

def test_files():
    surveyor = Surveyor()
    sample_query = 'quantum entanglement'
    zip_file, survey_file = surveyor.survey(sample_query, max_search=10, num_papers=6,
                    debug=False, weigh_authors=False)
    assert os.path.exists(zip_file)
    assert os.path.exists(survey_file)
