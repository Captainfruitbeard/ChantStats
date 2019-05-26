import os
from chantstats.v2.repertoire_and_genre import RepertoireAndGenreType
from chantstats.v2.analysis_functions import AnalysisType
from chantstats.v2.unit import UnitType


class ResultDescriptor:
    def __init__(self, rep_and_genre, analysis, unit, modal_category):
        self.rep_and_genre = RepertoireAndGenreType(rep_and_genre)
        self.analysis = AnalysisType(analysis)
        self.unit = UnitType(unit)
        self.modal_category = modal_category

        self.sep = "|"
        self.output_dirname = os.path.join(
            self.rep_and_genre.output_path_stub_1,
            self.analysis.output_path_stub,
            self.rep_and_genre.output_path_stub_2,
            self.unit.output_path_stub,
        )

    def get_full_output_path(self, filename_suffix):
        output_filename = self.sep.join(
            [
                # self.modal_category.output_path_stub_1,
                self.modal_category.output_path_stub_2,
                filename_suffix,
            ]
        )
        return os.path.join(self.output_dirname, output_filename)

    def __repr__(self):
        result_descriptor_stubs = (
            self.rep_and_genre.output_path_stub_1,
            self.analysis.output_path_stub,
            self.rep_and_genre.output_path_stub_2,
            self.unit.output_path_stub,
            self.modal_category.output_path_stub_1,
            self.modal_category.output_path_stub_2,
        )
        return f"<{result_descriptor_stubs}>"
