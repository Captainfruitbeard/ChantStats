import music21
import os
import pandas as pd
import re
from functools import lru_cache
from glob import glob
from tqdm import tqdm
from time import time
from music21.note import Note

from ..logging import logger
from ..pitch_class import PC
from ..repertoire_and_genre import RepertoireAndGenreType
from .helpers import group_by_contiguous_values, pairwise
from .organum_piece_section import OrganumPieceSection
from .organum_phrase import OrganumPhrase
from .organum_purum_duplum_part import OrganumPurumDuplumPart
from ..analysis_functions import (
    calculate_L5_occurrences,
    calculate_L4_occurrences,
    calculate_M5_occurrences,
    calculate_M4_occurrences,
)


def warn_if_tenor_does_not_start_on_first_note_of_each_measure(df):
    df_first_row_of_each_group = df.groupby([("common", "measure")]).apply(lambda grp: grp.iloc[0])
    df_first_row_of_each_group = df_first_row_of_each_group[
        df_first_row_of_each_group[("common", "texture")] != "discant_or_copula"
    ]
    if df_first_row_of_each_group["tenor"].isnull().any().any():
        logger.warning(
            "There is a measure where the tenor does not start on the first note! Check if this causes problems..."
        )


def merge_series(s1, s2, dtype=None):
    """
    Merge two pandas series.

    If values at corresponding positions in s1, s2 are different from each other
    or if the result contains any null values then an error will be raised.
    """

    def merge_values(x, y):
        if pd.isnull(x):
            assert not pd.isnull(y)
            return y
        else:
            assert (x == y) or pd.isnull(y)
            return x

    result = s1.combine(s2, merge_values)
    if dtype:
        result = result.astype(dtype)
    return result


def is_single_measure(df):
    return len(df[("common", "measure")].unique()) == 1


def get_unique_column_value(df, col):
    vals = df[col].unique()
    assert len(vals) == 1, "Column value must be unique but got: {}".format(vals)
    return vals[0]


def has_compound_meter(df):
    assert is_single_measure(df)
    time_signature = get_unique_column_value(df, ("common", "time_signature"))
    compound_time_signature_pattern = "^[0-9]+/8$"
    if re.match(compound_time_signature_pattern, time_signature):
        return True
    else:
        return False


def get_texture(df):
    if has_compound_meter(df):
        return "discant_or_copula"
    elif df[("duplum", "note")].isnull().all():
        return "chant"
    else:
        return "organum_purum"


def add_texture(df):
    res = df.copy()
    res[("common", "texture")] = get_texture(df)
    return res


def calculate_dataframe_from_single_part_stream(stream):
    assert isinstance(stream, music21.stream.Part)

    def get_current_time_signature(note):
        return note.getContextByClass("TimeSignature").ratioString

    def get_lyric(note):
        return note.lyric if note.lyric is not None else ""

    def get_note_info(note):
        return (
            note.offset,
            note.name,
            # note.nameWithOctave,
            note,
            note.pitch.ps,
            note.duration.quarterLength,
            note.measureNumber,
            get_current_time_signature(note),
            get_lyric(note),
        )

    columns = ["offset", "pitch_class", "note", "pitch", "duration", "measure", "time_signature", "lyric"]
    df = pd.DataFrame([get_note_info(n) for n in stream.flat.notes], columns=columns)
    return df.set_index("offset")


def merge_corresponding_columns(df, colname, dtype=None):
    df[("common", colname)] = merge_series(df[("tenor", colname)], df[("duplum", colname)], dtype=dtype)
    del df[("tenor", colname)]
    del df[("duplum", colname)]


# TODO: to make things more efficient, avoid all this iteration over streams!
def get_stanza_boundary_offsets(stream):
    duplum_barlines = list(stream.parts[0].recurse(classFilter="Barline"))
    tenor_barlines = list(stream.parts[1].recurse(classFilter="Barline"))

    # sanity check that barlines in duplum and tenor parts coincide
    assert [(b.offset, b.style) for b in duplum_barlines] == [(b.offset, b.style) for b in tenor_barlines]

    barline_offsets = []
    for b in duplum_barlines:
        assert b.style in ["double", "final"], "Unexpected barline type: '{}' (measure: {})".format(
            b.style, b.measureNumber
        )
        if b.style == "final":
            offset = b.getOffsetInHierarchy(stream.parts[0])
            logger.debug("Found barline '{}' at offset {}, measure {}".format(b.style, offset, b.measureNumber))
            barline_offsets.append(offset)
    barline_offsets.insert(0, 0.0)

    return barline_offsets


def calculate_dataframe_from_music21_stream(stream, filename, descr_stub):
    """
    Given a music21 Stream, return a pandas DataFrame representing it.
    """
    assert len(stream.parts) == 2

    duplum = stream.parts[0]
    tenor = stream.parts[1]

    df_duplum = calculate_dataframe_from_single_part_stream(duplum)
    df_tenor = calculate_dataframe_from_single_part_stream(tenor)

    df = pd.concat([df_tenor, df_duplum], axis=1, keys=["tenor", "duplum"])
    merge_corresponding_columns(df, "measure", dtype=int)
    merge_corresponding_columns(df, "time_signature")

    # Sanity check for contiguous range of measures
    measures_expected = list(range(1, df[("common", "measure")].max() + 1))
    measures_found = list(df[("common", "measure")].unique())
    if measures_expected != measures_found:
        logger.warning(
            f"Dataframe for piece {filename} is missing measures between 1-{max(measures_expected)}: "
            f"{sorted(set(measures_expected).difference(measures_found))}"
        )

    # df['tenor'] = forward_fill_each_measure(df['tenor'])
    # assert not df[('tenor', 'measure')].isnull().any()
    # assert not df[('tenor', 'time_signature')].isnull().any()

    # df.groupby([('common', 'measure')]).apply(add_texture)
    # df[('common', 'texture')] = pd.concat([add_texture(x) for _, x in df.groupby([('common', 'measure')])])
    ggg = df.groupby([("common", "measure")])
    df = pd.concat([add_texture(x) for _, x in ggg], axis=0)

    note_of_chant_final = df.iloc[-1]["tenor"]["note"]

    #
    # Fill phrase numbers
    #
    df[("common", "phrase")] = None

    ### OLD CODE. This relies on OrganumPieceSection, which is not great.
    # phrase_num = 0
    # for _, df_grp in group_by_contiguous_values(df, ("common", "texture")):
    #     sec = OrganumPieceSection(df_grp, filename, descr_stub, note_of_chant_final)
    #     for phrase in sec.phrases:
    #         phrase_num += 1
    #         df.loc[phrase.df.index, ("common", "phrase")] = phrase_num

    ### NEW CODE: self-contained and cleaner
    phrase_num = 0
    for _, df_texture in group_by_contiguous_values(df, ("common", "texture")):
        cur_texture = df_texture["common", "texture"].unique()[0]
        if cur_texture != "organum_purum":
            continue

        tenor_ffilled = df_texture["tenor"].ffill().dropna()
        for i, df_phrase in group_by_contiguous_values(df_texture, tenor_ffilled["note"]):
            phrase_num += 1
            # tenor_note = df_phrase["tenor", "note"].iloc[0]
            # print(f"Filling in phrase: {phrase_num} (tenor note: {tenor_note})")
            # print(f"   {df_phrase.index[0]}-{df_phrase.index[-1]}")
            df.loc[df_phrase.index, ("common", "phrase")] = phrase_num

    #
    # Calculate stanzas
    #
    stanza_boundary_offsets = get_stanza_boundary_offsets(stream)

    # sanity check that last barline was found at the end of the piece (after all notes)
    assert all(df.index < stanza_boundary_offsets[-1])

    df[("common", "stanza")] = None
    for stanza_num, (start_offset, end_offset) in enumerate(pairwise(stanza_boundary_offsets), start=1):
        # TODO: this raises a pandas warning because we're altering a copy; rewrite this in a safer way!
        df.loc[(start_offset <= df.index) & (df.index < end_offset), ("common", "stanza")] = stanza_num

    # verify that all entries have been assigned to some stanza
    assert not df[("common", "stanza")].isnull().any()

    df = df[["common", "tenor", "duplum"]]  # rearrange columns

    # TODO: this is just for sanity checking because it might cause us trouble
    warn_if_tenor_does_not_start_on_first_note_of_each_measure(df)

    # Check that the last measure has texture 'chant'
    assert df[("common", "texture")].iloc[-1] == "chant"

    return df


class OrganumPiece:
    def __init__(self, stream_or_filename):
        if isinstance(stream_or_filename, str):
            self.stream = music21.converter.parse(stream_or_filename)
            self.filename_full = os.path.abspath(stream_or_filename)
            self.filename_short = os.path.basename(stream_or_filename)
        # elif isinstance(stream_or_filename, music21.stream.Stream):
        #     self.stream = stream_or_filename
        #     self.filename_full = ""
        #     self.filename_short = ""
        else:  # pragma: no cover
            raise TypeError(f"Cannot load piece from object of type: '{type(stream_or_filename)}'")

        # TODO: extract stub_descr from filename
        self.descr_stub = re.match("^(F3.*)\.xml$", self.filename_short).group(1)

        self.df = calculate_dataframe_from_music21_stream(self.stream, self.filename_short, self.descr_stub)
        self.note_of_chant_final = self.df.iloc[-1]["tenor"]["note"]
        assert isinstance(self.note_of_chant_final, Note)
        self.note_of_final = self.note_of_chant_final  # alias
        self.chant_final = PC.from_note(self.note_of_final)
        self.final = self.chant_final  # alias
        self.organum_purum_duplum_part = OrganumPurumDuplumPart(self)

    def __repr__(self):
        return "<OrganumPiece: '{}'>".format(self.descr_stub)

    @property
    def descr(self):
        return self.filename_short

    @property
    def phrases(self):
        """
        Iterate over all phrases in this OrganumPiece
        """
        return [
            OrganumPhrase(df_phrase, self.filename_short, self.descr_stub)
            for _, df_phrase in self.df.dropna(subset=[("common", "phrase")]).groupby([("common", "phrase")])
        ]

    @property
    def sections(self):
        sections = []
        for _, df_grp in group_by_contiguous_values(self.df, ("common", "texture")):
            sections.append(OrganumPieceSection(df_grp, self.filename_short, self.descr_stub, self.note_of_chant_final))
        return sections

    def get_occurring_mode_degrees(self):
        return set(self.organum_purum_duplum_part.mode_degrees)

    def _get_L_or_M_occurrences(self, which, unit):
        funcs = {
            "L5": calculate_L5_occurrences,
            "L4": calculate_L4_occurrences,
            "M5": calculate_M5_occurrences,
            "M4": calculate_M4_occurrences,
        }
        return set(funcs[which](self.organum_purum_duplum_part, unit=unit))


@lru_cache(maxsize=10)
def load_organum_pieces(input_dir, *, pattern="*.xml"):
    """
    Load responsorial chant pieces from MusicXML files in a given input directory.

    Parameters
    ----------
    input_dir : str
        Input directory in which to look for MusicXML files.
    pattern : str, optional
        Filename pattern; this can be used to filter the files
        to be loaded to a subset (for example during testing).

    Returns
    -------
    list of ResponsorialChantPiece
    """
    pattern = pattern if pattern is not None else "*.xml"
    filenames = sorted(glob(os.path.join(input_dir, pattern)))
    logger.debug(f"Found {len(filenames)} pieces matching the pattern '{pattern}'.")
    logger.debug(f"Loading pieces... ")
    tic = time()
    pieces = [OrganumPiece(f) for f in tqdm(filenames)]
    toc = time()
    logger.debug(f"Done. Loaded {len(pieces)} pieces.")
    logger.debug(f"Loading pieces took {toc-tic:.2f} seconds.")
    return pieces


class OrganumPieces:
    def __init__(self, pieces):
        assert all([isinstance(p, OrganumPiece) for p in pieces])
        self.pieces = pieces
        self.repertoire_and_genre = RepertoireAndGenreType("organum_pieces")

    def __repr__(self):
        return f"<Collection of {len(self.pieces)} organum pieces>"

    def __getitem__(self, idx):
        return self.pieces[idx]

    def __iter__(self):
        yield from self.pieces

    @classmethod
    def from_musicxml_files(cls, cfg, filename_pattern=None):
        musicxml_path = cfg.get_musicxml_path("organum_pieces")
        pieces = load_organum_pieces(musicxml_path, pattern=filename_pattern)
        return cls(pieces)

    def get_analysis_inputs(
        self,
        mode=None,
        min_num_phrases_per_monomodal_section=None,
        min_num_notes_per_monomodal_section=None,
        min_num_notes_per_organum_phrase=None,
    ):
        return [piece.organum_purum_duplum_part for piece in self.pieces]

    def get_occurring_mode_degrees(self):
        mds = set()
        for piece in self.pieces:
            mds.update(piece.get_occurring_mode_degrees())
        return mds

    def get_L_and_M_occurrences(self, which, unit):
        res = set()
        if which == "L5M5":
            res.update(self._get_L_or_M_occurrences("L5", unit))
            res.update(self._get_L_or_M_occurrences("M5", unit))
        elif which == "L4M4":
            res.update(self._get_L_or_M_occurrences("L4", unit))
            res.update(self._get_L_or_M_occurrences("M4", unit))
        else:
            raise NotImplementedError()
        return res

    def _get_L_or_M_occurrences(self, which, unit):
        funcs = {
            "L5": calculate_L5_occurrences,
            "L4": calculate_L4_occurrences,
            "M5": calculate_M5_occurrences,
            "M4": calculate_M4_occurrences,
        }
        res = set()
        for piece in self.pieces:
            res.update(piece._get_L_or_M_occurrences(which, unit))
        return res


class OrganumPhrases:
    def __init__(self, phrases):
        assert all([isinstance(p, OrganumPhrase) for p in phrases])
        self.phrases = phrases
        self.repertoire_and_genre = RepertoireAndGenreType("organum_phrases")

    def __repr__(self):
        return f"<Collection of {len(self.phrases)} organum phrases>"

    def __getitem__(self, idx):
        return self.phrases[idx]

    def __iter__(self):
        yield from self.phrases

    @classmethod
    def from_musicxml_files(cls, cfg, filename_pattern=None):
        musicxml_path = cfg.get_musicxml_path("organum_pieces")
        pieces = load_organum_pieces(musicxml_path, pattern=filename_pattern)
        phrases = sum([piece.phrases for piece in pieces], [])
        return cls(phrases)

    def get_analysis_inputs(
        self,
        mode=None,
        min_num_phrases_per_monomodal_section=None,
        min_num_notes_per_monomodal_section=None,
        min_num_notes_per_organum_phrase=14,
        max_num_notes_per_organum_phrase=19,
    ):
        return [p for p in self.phrases if len(p.notes) >= min_num_notes_per_organum_phrase]

    def get_occurring_mode_degrees(self):
        mds = set()
        for phrase in self.phrases:
            mds.update(phrase.mode_degrees)
        return mds
