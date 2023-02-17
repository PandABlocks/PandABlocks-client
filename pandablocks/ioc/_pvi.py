import json
import logging
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List

from epicsdbbuilder import RecordName
from pvi._format.dls import DLSFormatter
from pvi.device import (
    ComboBox,
    Component,
    Device,
    DeviceRef,
    Grid,
    Group,
    Row,
    SignalR,
    SignalRW,
    SignalX,
    TextRead,
    TextWrite,
    Tree,
)
from softioc import builder
from softioc.pythonSoftIoc import RecordWrapper

from pandablocks.ioc._types import OUT_RECORD_FUNCTIONS, EpicsName


class PviGroup(Enum):
    """Categories to group record display widgets"""

    NONE = None  # This marks a top-level group
    INPUTS = "Inputs"
    PARAMETERS = "Parameters"
    READBACKS = "Readbacks"
    OUTPUTS = "Outputs"
    TABLE = "Table"  # TODO: May not need this anymore


@dataclass
class PviInfo:
    """A container for PVI related information for a record

    `group`: The group that this info is a part of
    `component`: The PVI Component used for rendering"""

    group: PviGroup
    component: Component


def add_pvi_info(
    group: PviGroup,
    record: RecordWrapper,
    record_name: EpicsName,
    record_creation_func: Callable,
) -> None:
    """Create the most common forms of the `PviInfo` structure"""
    component: Component
    writeable: bool = record_creation_func in OUT_RECORD_FUNCTIONS
    useComboBox: bool = record_creation_func == builder.mbbOut

    if record_creation_func == builder.Action:
        # TODO: What value do I write? PandA uses an empty string
        component = SignalX(record_name, record_name, value="")
        access = "x"
    elif writeable:
        if useComboBox:
            widget = ComboBox()
        else:
            widget = TextWrite()
        component = SignalRW(record_name, record_name, widget)
        access = "rw"
    else:
        component = SignalR(record_name, record_name, TextRead())
        access = "r"
    block, field = record_name.split(":", maxsplit=1)
    record.add_info(
        "Q:group",
        {
            RecordName(f"{block}:PVI"): {
                f"pvi.{field.replace(':', '_')}.{access}": {
                    "+channel": "NAME",
                    "+type": "plain",
                }
            }
        },
    )
    Pvi.add_pvi_info(record_name=record_name, group=group, component=component)


_positions_table_group = Group("POSITIONS_TABLE", Grid(labelled=True), children=[])
_positions_columns_defs = [
    # TODO: To exactly copy the PandA Table web GUI, we'll need a new widget
    # type that displays a static string in the same space as a PV
    # ("NAME", record_name),
    ("VALUE", SignalR),
    ("UNITS", SignalRW),
    ("SCALE", SignalRW),
    ("OFFSET", SignalRW),
    ("CAPTURE", SignalRW),
]


# TODO: Replicate this for the BITS table
def add_positions_table_row(
    record_name: str,
    units_record_name: str,
    scale_record_name: str,
    offset_record_name: str,
    capture_record_name: str,
) -> None:
    """Add a Row to the Positions table"""
    # TODO: Use the Components defined in _positions_columns_defs to
    # create the children, which will make it more obvious which
    # component is for which column
    children = [
        SignalR(record_name, record_name, TextRead()),
        SignalRW(units_record_name, units_record_name, TextWrite()),
        SignalRW(scale_record_name, scale_record_name, TextWrite()),
        SignalRW(offset_record_name, offset_record_name, TextWrite()),
        SignalRW(capture_record_name, capture_record_name, TextWrite()),
    ]

    row = Row()
    if len(_positions_table_group.children) == 0:
        row.header = [k[0] for k in _positions_columns_defs]

    row_group = Group(
        record_name + "_row",
        row,
        children,
    )

    _positions_table_group.children.append(row_group)


class Pvi:
    """TODO: Docs"""

    # pvi_info_dict: Dict[EpicsName, PviInfo] = {}
    pvi_info_dict: Dict[str, Dict[PviGroup, List[Component]]] = {}
    bob_file_dict: Dict[str, str] = {}

    @staticmethod
    def add_pvi_info(record_name: EpicsName, group: PviGroup, component: Component):
        """Add PVI Info to the global collection"""

        record_base, _ = record_name.split(":", 1)

        if record_base in Pvi.pvi_info_dict:
            if group in Pvi.pvi_info_dict[record_base]:
                Pvi.pvi_info_dict[record_base][group].append(component)
            else:
                Pvi.pvi_info_dict[record_base][group] = [component]
        else:
            Pvi.pvi_info_dict[record_base] = {group: [component]}

    @staticmethod
    def create_pvi_records(record_prefix: str):
        """Create the :PVI records, one for each block and one at the top level"""

        devices: List[Device] = []
        pvi_records: List[str] = []
        for block_name, v in Pvi.pvi_info_dict.items():
            children: Tree = []

            # Item in the NONE group should be rendered outside of any Group box
            if PviGroup.NONE in v:
                children.extend(v.pop(PviGroup.NONE))
            for group, components in v.items():
                children.append(Group(group.name, Grid(), components))

            device = Device(block_name, children)
            devices.append(device)

            # Add PVI structure. Unfortunately we need something in the database
            # that holds the PVI PV, and the QSRV records we have made so far aren't in the
            # database, so have to make an extra record here just to hold the PVI PV name
            pvi_record_name = block_name + ":PVI"
            block_pvi = builder.stringIn(
                pvi_record_name + "_PV", initial_value=RecordName(pvi_record_name)
            )
            block_pvi.add_info(
                "Q:group",
                {
                    RecordName(f"PVI"): {
                        f"pvi.{block_name}.d": {
                            "+channel": "VAL",
                            "+type": "plain",
                        }
                    }
                },
            )

            pvi_records.append(pvi_record_name)

        # TODO: Properly add this to list of screens, add a PV, maybe roll into
        # the "PLACEHOLDER" Device?
        # Add Tables to a new top level screen
        top_device = Device("PandA", children=[_positions_table_group])
        devices.append(top_device)

        # Create top level Device, with references to all child Devices
        device_refs = [DeviceRef(x, x) for x in pvi_records]

        # # TODO: What should the label be?
        device = Device("TOP", device_refs)
        devices.append(device)

        # TODO: label widths need some tweaking - some are pretty long right now
        formatter = DLSFormatter(label_width=250)
        with tempfile.TemporaryDirectory() as temp_dir:
            for device in devices:
                try:
                    formatter.format(
                        device,
                        record_prefix + ":",
                        Path(f"{temp_dir}/{device.label}.bob"),
                    )
                    with open(f"{temp_dir}/{device.label}.bob") as f:
                        Pvi.bob_file_dict.update({f"{device.label}.bob": f.read()})

                except NotImplementedError:

                    logging.exception("Cannot create TABLES yet")
