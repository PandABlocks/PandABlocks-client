from pandablocks.ioc._types import (
    EpicsName,
    PandAName,
    _epics_to_panda_name,
    _panda_to_epics_name,
)


def test_panda_to_epics_name_conversion() -> None:
    assert _panda_to_epics_name(PandAName("ABC.123.456")) == EpicsName("ABC:123:456")


def test_epics_to_panda_name_conversion() -> None:
    assert _epics_to_panda_name(EpicsName("ABC:123:456")) == PandAName("ABC.123.456")


def test_panda_to_epics_and_back_name_conversion() -> None:
    """Test panda->EPICS->panda round trip name conversion works"""
    assert _epics_to_panda_name(
        _panda_to_epics_name(PandAName("ABC.123.456"))
    ) == PandAName("ABC.123.456")
