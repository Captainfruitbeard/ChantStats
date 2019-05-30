import os
import pytest
from approvaltests.approvals import verify

from .context import chantstats
from chantstats.v2 import ChantStatsConfig, calculate_results, export_results, logger, load_pieces
from chantstats.v2.utils import list_directory_tree


@pytest.mark.slow
def test_folder_structure_for_exported_results(tmpdir, diff_reporter):
    output_root_dir = str(tmpdir)
    logger.info(f"Using output root dir: '{output_root_dir}'")

    cfg = ChantStatsConfig.from_env()
    plainchant_sequence_pieces = load_pieces("plainchant_sequences", cfg)
    responsorial_chant_pieces = load_pieces("responsorial_chants", cfg, filename_pattern="F3M[0-1]*.xml")

    for pieces in [plainchant_sequence_pieces, responsorial_chant_pieces]:
        results_pc_freqs = calculate_results(
            pieces=pieces,
            analysis="pc_freqs",
            sampling_fraction=1.0,
            sampling_seed=None,
            min_num_phrases_per_monomodal_section=3,
            min_num_notes_per_monomodal_section=80,
            modes=["final"],
            units=["pcs", "mode_degrees"],
            modal_category_keys=["C"],
        )
        results_tendency = calculate_results(
            pieces=pieces,
            analysis="tendency",
            sampling_fraction=1.0,
            sampling_seed=None,
            min_num_phrases_per_monomodal_section=3,
            min_num_notes_per_monomodal_section=80,
            modes=["final_and_ambitus"],
            units=["pcs", "mode_degrees"],
            modal_category_keys=[("E", "authentic")],
        )
        results_L5M5 = calculate_results(
            pieces=pieces,
            analysis="L_and_M__L5_u_M5",
            sampling_fraction=1.0,
            sampling_seed=None,
            min_num_phrases_per_monomodal_section=3,
            min_num_notes_per_monomodal_section=80,
            modes=["final_and_ambitus"],
            units=["pcs", "mode_degrees"],
            modal_category_keys=[("E", "authentic")],
        )
        results_L4M4 = calculate_results(
            pieces=pieces,
            analysis="L_and_M__L4_u_M4",
            sampling_fraction=1.0,
            sampling_seed=None,
            min_num_phrases_per_monomodal_section=3,
            min_num_notes_per_monomodal_section=80,
            modes=["final_and_ambitus"],
            units=["pcs", "mode_degrees"],
            modal_category_keys=[("E", "authentic")],
        )
        export_results(results_pc_freqs, output_root_dir, p_cutoff=0.4)
        export_results(results_tendency, output_root_dir, p_cutoff=0.4)
        export_results(results_L5M5, output_root_dir, p_cutoff=0.4)
        export_results(results_L4M4, output_root_dir, p_cutoff=0.4)

    exported_files = list_directory_tree(output_root_dir)
    verify(exported_files, diff_reporter)


@pytest.mark.slow
def test_run_analyses_and_export_results(tmpdir):
    output_root_dir = os.getenv("CHANTSTATS_OUTPUT_ROOT_DIR", str(tmpdir))
    logger.info(f"Using output root dir: '{output_root_dir}'")

    cfg = ChantStatsConfig.from_env()
    repertoires_and_genres = ["plainchant_sequences", "responsorial_chants"]
    p_cutoff = 0.4
    sampling_fraction = 0.7
    sampling_seed = 99999
    min_num_phrases_per_monomodal_section = 3
    min_num_notes_per_monomodal_section = 80

    for rep_and_genre in repertoires_and_genres:
        pieces = load_pieces(rep_and_genre, cfg)

        for analysis in ["pc_freqs", "tendency", "L_and_M__L5_u_M5", "L_and_M__L4_u_M4"]:
            logger.info(f"Calculating results for analysis '{analysis}'")
            results = calculate_results(
                pieces=pieces,
                analysis=analysis,
                sampling_fraction=sampling_fraction,
                sampling_seed=sampling_seed,
                min_num_phrases_per_monomodal_section=min_num_phrases_per_monomodal_section,
                min_num_notes_per_monomodal_section=min_num_notes_per_monomodal_section,
            )

            logger.info(f"Exporting results for analysis '{analysis}'")
            export_results(results, output_root_dir, p_cutoff=p_cutoff)
