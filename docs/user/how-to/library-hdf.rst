.. _library-hdf:

How to use the library to capture HDF files
===========================================

The `commandline-hdf` introduced how to use the commandline to capture HDF files.
The `write_hdf_files` function that is called to do this can also be integrated
into custom Python applications. This guide shows how to do this.

Approach 1: Call the function directly
--------------------------------------

If you need a one-shot configure and run application, you can use the
function directly:

.. literalinclude:: ../../../examples/arm_and_hdf.py

With the `AsyncioClient` as a `Context Manager <typecontextmanager>`, this code
sets up some fields of a PandA before taking a single acquisition. The code in
`write_hdf_files` is responsible for arming the PandA.

.. note::

    There are no log messages emitted like in `commandline-hdf`. This is because
    we have not configured the logging framework in this example. You can get
    these messages by adding a call to `logging.basicConfig` like this::

        logging.basicConfig(level=logging.INFO)

Approach 2: Create the pipeline yourself
----------------------------------------

If you need more control over the pipeline, for instance to display progress,
you can create the pipeline yourself, and feed it data from the PandA. This
means you can make decisions about when to start and stop acquisitions based on
the `Data` objects that go past. For example, if we want to make a progress bar
we could:

.. literalinclude:: ../../../examples/hdf_queue_reporting.py

This time, after setting up the PandA, we create the `AsyncioClient.data`
iterator ourselves. Each `Data` object we get is queued on the first `Pipeline`
element, then inspected. The type of object tells us if we should Arm the PandA,
update a progress bar, or return as acquisition is complete.

In a `finally <finally>` block we stop the pipeline, which will wait for all data
to flow through the pipeline and close the HDF file.

Performance
-----------

The commandline client and both these approaches use the same core code, so will
give the same performance. The steps to consider in optimising performance are
outlined in `performance`