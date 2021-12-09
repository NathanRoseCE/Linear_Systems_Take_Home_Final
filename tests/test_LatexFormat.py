from scripts import LatexFormat


def test_RoundFloat():
    assert "4.312" == LatexFormat.round_float(4.3116)


def test_imaginary():
    assert r"3.123 \pm 9.324j" == LatexFormat.imaginary({
        "real": 3.1229,
        "imaginary": 9.3244
    })
