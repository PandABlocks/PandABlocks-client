

def seq_table2cmd(repeats: int, trigger: int, position: int, time1: int, phase1: dict, time2: int, phase2: dict):
    """
    Function to encode (repeats, trigger, phase1, phase2) into 32bit integer (code).

    Return a list of str [code, position, time1, time2] to send to pandabox sequencer (SEQ)
    table.

    Example:
        ```python
        keys = ["a", "b", "c", "d", "e", "f"]  
        repeats = 1 
        time1 = 0  
        position = 0  
        trigger = "Immediate"  
        phase1 = {key: False for key in keys} 
        time2 = 0  
        phase2 = {key: False if key != "a" else True for key in keys}
        pandabox.send(
                Put(
                    "SEQ1.TABLE",seq_table(
                        repeats, trigger, position, time1, phase2, time2, phase2
                    ),
                )
            )
        ```
    """
    trigger_options = {
        "Immediate": 0,
        "bita=0": 1,
        "bita=1": 2,
        "bitb=0": 3,
        "bitb=1": 4,
        "bitc=0": 5,
        "bitc=1": 6,
        "posa>=position": 7,
        "posa<=position": 8,
        "posb>=position": 9,
        "posb<=position": 10,
        "posc>=position": 11,
        "posc<=position": 12,
    }

    # _b binary code
    repeats_b = "{0:016b}".format(repeats)  # 16 bits
    trigger_b = "{0:04b}".format(trigger_options[trigger])  # 4 bits (17-20)
    phase1_b = ""
    for key, value in sorted(phase1.items()):  # 6 bits (a-f)
        phase1_b = "1" + phase1_b if value else "0" + phase1_b
    phase2_b = ""
    for key, value in sorted(phase2.items()):  # 6 bits (a-f)
        phase2_b = "1" + phase2_b if value else "0" + phase2_b
    code_b = phase2_b + phase1_b + trigger_b + repeats_b  # 32 bits code
    code = int(code_b, 2)

    # a table line = [code position time1 time2]
    pos_cmd = [
        f"{code:d}",
        f"{int(position):d}",
        f"{int(time1):d}",
        f"{int(time2):d}",
    ]

    return pos_cmd
