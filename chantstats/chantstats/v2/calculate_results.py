from .analysis_type import AnalysisType
from .dendrogram import calculate_dendrogram
from .logging import logger
from .modal_category import ModalCategoryType, GroupingByModalCategory
from .plainchant_sequence_piece import PlainchantSequencePieces
from .repertoire_and_genre import RepertoireAndGenreType
from .result_descriptor import ResultDescriptor
from .unit import UnitType
from .utils import get_subsample

__all__ = ["calculate_results"]


class PathStubs(tuple):
    def __new__(cls, repertoire_and_genre, analysis, unit, modal_category):
        rep_and_genre = RepertoireAndGenreType(repertoire_and_genre)
        analysis = AnalysisType(analysis)
        unit = UnitType(unit)
        modal_category = modal_category

        path_stubs = (
            rep_and_genre.output_path_stub_1,
            analysis.output_path_stub,
            rep_and_genre.output_path_stub_2,
            unit.output_path_stub,
            modal_category.output_path_stub_1,
            modal_category.output_path_stub_2,
        )

        obj = tuple.__new__(cls, path_stubs)
        obj.rep_and_genre = rep_and_genre
        obj.analysis = analysis
        obj.unit = unit
        obj.modal_category = modal_category
        obj.path_stubs = path_stubs
        obj.path_stubs_without_unit_and_mode = path_stubs[:3]

        return obj


def calculate_results(
    *,
    pieces,
    analysis,
    sampling_fraction,
    sampling_seed,
    min_num_phrases_per_monomodal_section=3,
    min_num_notes_per_monomodal_section=80,
    modes=None,
    units=None,
    modal_category_keys=None,
):
    assert isinstance(pieces, PlainchantSequencePieces)
    modes = modes or list(ModalCategoryType)
    units = units or list(UnitType)

    results = {}
    for mode in modes:
        analysis_inputs = pieces.get_analysis_inputs(
            mode,
            min_num_phrases_per_monomodal_section=min_num_phrases_per_monomodal_section,
            min_num_notes_per_monomodal_section=min_num_notes_per_monomodal_section,
        )
        analysis_inputs_subsample = get_subsample(analysis_inputs, sampling_fraction, seed=sampling_seed)

        grouping = GroupingByModalCategory(analysis_inputs_subsample, group_by=mode)
        keys = modal_category_keys or grouping.keys
        for key in keys:
            modal_category = grouping[key]
            logger.info(f"Calculating {analysis} results for {modal_category}")
            for unit in units:
                dendrogram = calculate_dendrogram(modal_category, analysis=analysis, unit=unit)
                result_descriptor = ResultDescriptor(pieces.repertoire_and_genre, analysis, unit, modal_category)
                results[result_descriptor] = {"dendrogram": dendrogram}

    return results
