from pathlib import Path
from sys import path as sys_path

src_directory = Path(__file__).resolve().parents[1].joinpath("src")
sys_path.append(str(src_directory))

from dataset import Marker
from experiment import Experiment, ExperimentCSVHandler


def test_update_donor():
    exp = Experiment()
    exp.update("21/03/2026 12:41:05 INFO:Donor: VGG16")
    assert exp.donor == "VGG16"


def test_update_unknown_field_ignored():
    exp = Experiment(macro_f1_per_class=["" for i in range(len(Marker))])
    exp.update("21/03/2026 12:41:05 INFO:Start testing")
    assert all(len(value) == 0 for _, value in exp)


def test_update_macro_f1_per_class_parsed_as_list():
    exp = Experiment()
    exp.update("21/03/2026 12:41:05 INFO:Macro f1 per class: [0.91, 0.87, 0.93]")
    assert exp.macro_f1_per_class == ["0.91", "0.87", "0.93"]


def test_iter_for_list_field():
    exp = Experiment(macro_f1_per_class=[f"{i / 10}" for i in range(len(Marker))])
    d = dict(exp)
    assert d["macro_f1_class_0"] == "0.0"
    assert d["macro_f1_class_1"] == "0.1"
    assert d["macro_f1_class_2"] == "0.2"
    assert d["macro_f1_class_3"] == "0.3"
    assert d["macro_f1_class_4"] == "0.4"
    assert d["macro_f1_class_5"] == "0.5"
    assert d["macro_f1_class_6"] == "0.6"
    assert d["macro_f1_class_7"] == "0.7"
    assert d["macro_f1_class_8"] == "0.8"
    assert d["macro_f1_class_9"] == "0.9"
    assert d["macro_f1_class_10"] == "1.0"
    assert d["macro_f1_class_11"] == "1.1"
    assert d["macro_f1_class_12"] == "1.2"
    assert d["macro_f1_class_13"] == "1.3"
    assert d["macro_f1_class_14"] == "1.4"


def test_headers_match_iter_order():
    exp = Experiment(
        donor="X",
        accuracy="0.9",
        macro_f1_per_class=[f"{i / 10}" for i in range(len(Marker))],
    )
    assert list(dict(exp).keys()) == Experiment.header()


def test_handler_creates_file_with_header(tmp_path: Path):
    path = tmp_path.joinpath("results.csv")
    exp = Experiment(
        donor="ABC", macro_f1_per_class=[f"{i / 10}" for i in range(len(Marker))]
    )
    with ExperimentCSVHandler(path) as handler:
        handler.writerow(exp)
    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(Experiment.header())
    assert "ABC" in lines[1]
    assert "0.0" in lines[1]
    assert "0.1" in lines[1]
    assert "0.2" in lines[1]
    assert "0.3" in lines[1]
    assert "0.4" in lines[1]
    assert "0.5" in lines[1]
    assert "0.6" in lines[1]
    assert "0.7" in lines[1]
    assert "0.8" in lines[1]
    assert "0.9" in lines[1]
    assert "1.0" in lines[1]
    assert "1.1" in lines[1]
    assert "1.2" in lines[1]
    assert "1.3" in lines[1]
    assert "1.4" in lines[1]


def test_handler_appends_without_duplicate_header(tmp_path: Path):
    path = tmp_path.joinpath("results.csv")
    exp = Experiment(
        donor="ABC", macro_f1_per_class=[f"{i / 10}" for i in range(len(Marker))]
    )
    with ExperimentCSVHandler(path) as handler:
        handler.writerow(exp)
    with ExperimentCSVHandler(path) as handler:
        handler.writerow(exp)
    lines = path.read_text().splitlines()
    header_count = sum(1 for line in lines if line.startswith("donor"))
    assert header_count == 1


def test_handler_write_many(tmp_path: Path):
    path = tmp_path.joinpath("results.csv")
    experiments = [
        Experiment(
            donor=f"D{i}", macro_f1_per_class=[f"{j / 10}" for j in range(len(Marker))]
        )
        for i in range(3)
    ]
    with ExperimentCSVHandler(path) as handler:
        handler.writerows(experiments)
    lines = path.read_text().splitlines()
    assert len(lines) == 4
