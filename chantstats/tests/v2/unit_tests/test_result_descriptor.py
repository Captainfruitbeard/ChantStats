from .context import chantstats
from chantstats.v2.modal_category import ModalCategory
from chantstats.v2.result_descriptor import ResultDescriptor


def test_result_descriptor():
    modal_category = ModalCategory(items=None, modal_category_type="final", key="G")
    rd = ResultDescriptor("plainchant_sequences", "pc_freqs", "pcs", modal_category)
    assert rd.output_dirname == "chant/pc_freqs/sequences/pcs"
    assert rd.get_full_output_path("dendrogram.png") == "chant/pc_freqs/sequences/pcs/final_G|dendrogram.png"
    assert rd.get_full_output_path("quux.png") == "chant/pc_freqs/sequences/pcs/final_G|quux.png"

    modal_category = ModalCategory(items=None, modal_category_type="final_and_ambitus", key=("C", "plagal"))
    rd = ResultDescriptor("plainchant_sequences", "tendency", "mode_degrees", modal_category)
    assert rd.output_dirname == "chant/tendency/sequences/mode_degrees"
    assert rd.get_full_output_path("dendrogram.png") == "chant/tendency/sequences/mode_degrees/plagal_C|dendrogram.png"
    assert rd.get_full_output_path("foobar.png") == "chant/tendency/sequences/mode_degrees/plagal_C|foobar.png"
