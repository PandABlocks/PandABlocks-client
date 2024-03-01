How to introspect a PandA
===========================

Using a combination of `commands <pandablocks.commands>` it is straightforward to query the PandA
to list all blocks, and all fields inside each block, that exist. 

Call the following script, with the address of the PandA as the first and only command line argument:


.. literalinclude:: ../../../examples/introspect_panda.py

This script can be found in ``examples/introspect_panda.py``.

By examining the `BlockInfo` structure returned from `GetBlockInfo` for each Block the number
and description may be acquired for every block.

By examining the `FieldInfo` structure (which is fully printed in this example) the ``type``, 
``sub-type``, ``description`` and ``label`` may all be found for every field. 

Lastly the complete list of every ``BITS`` field in the ``PCAP`` block are gathered and
printed. See the documentation in the `Field Types <https://pandablocks-server.readthedocs.io/en/latest/fields.html?#field-types>`_
section of the PandA Server documentation.
