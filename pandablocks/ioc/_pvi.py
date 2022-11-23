import json
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List

from pvi._format.dls import DLSFormatter
from pvi.device import (
    ComboBox,
    Component,
    Device,
    DeviceRef,
    Grid,
    Group,
    SignalR,
    SignalRW,
    SignalX,
    TextRead,
    TextWrite,
    Tree,
)
from softioc import builder

from pandablocks.ioc._types import OUT_RECORD_FUNCTIONS, EpicsName


class PviGroup(Enum):
    """Categories to group record display widgets"""

    NONE = None  # This marks a top-level group
    INPUTS = "Inputs"
    PARAMETERS = "Parameters"
    READBACKS = "Readbacks"
    OUTPUTS = "Outputs"
    TABLE = "Table"


@dataclass
class PviInfo:
    """A container for PVI related information for a record

    `group`: The group that this info is a part of
    `component`: The PVI Component used for rendering"""

    group: PviGroup
    component: Component


def add_pvi_info(
    group: PviGroup,
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
    elif writeable:
        if useComboBox:
            widget = ComboBox()
        else:
            widget = TextWrite()
        component = SignalRW(record_name, record_name, widget)
    else:
        component = SignalR(record_name, record_name, TextRead())

    Pvi.add_pvi_info(record_name=record_name, group=group, component=component)


class Pvi:
    """TODO: Docs"""

    # pvi_info_dict: Dict[EpicsName, PviInfo] = {}
    pvi_info_dict: Dict[str, Dict[PviGroup, List[Component]]] = {}

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
    def create_pvi_records():
        """Create the :PVI records, one for each block and one at the top level"""

        devices: List[Device] = []
        pvi_records: List[str] = []
        for block_name, v in Pvi.pvi_info_dict.items():
            children: Tree = []

            # Item in the NONE group should be rendered outside of any Group box
            children.extend(v.pop(PviGroup.NONE))
            for group, components in v.items():
                children.append(Group(group.name, Grid(), components))

            device = Device(block_name, children)
            devices.append(device)

            pvi_record_name = block_name + ":PVI"
            builder.longStringIn(
                pvi_record_name, initial_value=json.dumps(device.serialize())
            )
            pvi_records.append(pvi_record_name)

        # Create top level Device, with references to all child Devices
        device_refs = [DeviceRef(x, x) for x in pvi_records]

        # # TODO: What should the label be?
        device = Device("PLACEHOLDER", device_refs)

        data = json.dumps(device.serialize())

        # # Top level PVI record
        builder.longStringIn("PVI", initial_value=data)

        # TODO: Temp code to test generating the .bob file
        # TODO: label widths need some tweaking - some are pretty long right now
        formatter = DLSFormatter(label_width=250)
        from pathlib import Path

        for device in devices:
            try:
                formatter.format(
                    device,
                    "ABC" + ":",
                    Path(
                        f"/home/eyh46967/dev/PandABlocks-client/bob/{device.label}.bob"
                    ),
                )
            except NotImplementedError:
                import logging

                logging.exception("Cannot create TABLES yet")
