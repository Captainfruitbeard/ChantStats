import os
import sys

from chantstats.v2 import ChantStatsConfig, calculate_results, export_results, logger, load_pieces
from chantstats.v2.repertoire_and_genre import RepertoireAndGenreType


def run_analyses_and_export_results(rep_and_genre, *, output_root_dir, analyses=None):
    logger.info(f"Using output root dir: '{output_root_dir}'")

    cfg = ChantStatsConfig.from_env()
    p_cutoff = 0.4
    sampling_fraction = 0.7
    sampling_seed = 99999
    min_num_phrases_per_monomodal_section = 3
    min_num_notes_per_monomodal_section = 80
    min_num_notes_per_organum_phrase = 14
    max_num_notes_per_organum_phrase = 19

    pieces = load_pieces(rep_and_genre, cfg)
    analyses = analyses or ["pc_freqs", "tendency", "L_and_M__L5_u_M5", "L_and_M__L4_u_M4"]

    if rep_and_genre in ("plainchant_sequences", "responsorial_chants"):
        modes = ["final", "final_and_ambitus"]
    elif rep_and_genre in ("organum_pieces", "organum_phrases"):
        modes = ["final"]
    else:
        raise ValueError(f"Invalid rep_and_genre: '{rep_and_genre}'")

    for analysis in analyses:
        logger.info(f"Calculating results for analysis '{analysis}'")
        results = calculate_results(
            pieces=pieces,
            analysis=analysis,
            sampling_fraction=sampling_fraction,
            sampling_seed=sampling_seed,
            min_num_phrases_per_monomodal_section=min_num_phrases_per_monomodal_section,
            min_num_notes_per_monomodal_section=min_num_notes_per_monomodal_section,
            min_num_notes_per_organum_phrase=min_num_notes_per_organum_phrase,
            modes=modes,
        )

        logger.info(f"Exporting results for analysis '{analysis}'")
        export_results(results, output_root_dir, p_cutoff=p_cutoff)


if __name__ == "__main__":
    output_root_dir = os.environ["CHANTSTATS_OUTPUT_ROOT_DIR"]
    rep_and_genre = RepertoireAndGenreType(sys.argv[1])
    try:
        analyses = sys.argv[2].split(",")
    except IndexError:
        analyses = None
    run_analyses_and_export_results(rep_and_genre, output_root_dir=output_root_dir, analyses=analyses)
