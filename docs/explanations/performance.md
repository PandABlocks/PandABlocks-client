(performance)=

# How fast can we write HDF files?

There are many factors that affect the speed we can write HDF files. This article
discusses how this library addresses them and what the maximum data rate of a PandA is.

## Factors to consider

```{eval-rst}
.. list-table::
    :widths: 10 50

    * - Trigger frequency
      - Each trigger will send all the captured fields, so the higher the trigger
        frequency the more data is sent
    * - Fields captured
      - Both the number of fields captured and their capture types affect the data
        sent on each trigger
    * - Sample format and processing
      - Server side format and processing of each sample affects the data volume.
        Framed format is selected by this library, with raw or scaled data
        processing available.
    * - Network speed
      - 1 Gigabit ethernet should be used to maximise throughput
    * - File system speed
      - Some local disks and NFS mounts may not be fast enough to sustain
        maximum data rate.
    * - CPU load on the PandA
      - Excessive CPU load on the PandA, generated by extra TCP server clients
        or panda-webcontrol will reduce throughput
    * - Flush rate
      - Flushing data to disk to often will slow write speed
```

## Strategies to help

There are a number of strategies that help increase performance. These can be
combined to give the greatest benefit

### Average the data

Selecting the `Mean` capture mode will activate on-FPGA averaging of the
captured value. `Min` and `Max` can also be captured at the same time.
Capturing these rather than `Value` may allow you to lower the trigger
frequency while still providing enough information for data analysis

### Scale the data on the client

`AsyncioClient.data` and `BlockingClient.data` accept a `scaled` argument.
Setting this to False will transfer the raw unscaled data, allowing for up to
50% more data to be sent depending on the datatype of the field. You can
use the `StartData.fields` information to scale the data on the client.
The `write_hdf_files` function uses this approach.

### Remove the panda-webcontrol package

The measures above should get you to about 50MBytes/s, but if more clients
connect to the web GUI then this will drop. To increase the data rate to
60MBytes/s and improve stability you may want to remove the panda-webcontrol
zpkg.

### Flush about 1Hz

`AsyncioClient.data` accepts a `flush_period` argument. If given, it will
squash intermediate data frames together until this period expires, and only
then produce them. This means the numpy data blocks are larger and can be more
efficiently written to disk then flushed. The `write_hdf_files` function uses
this approach.

## Performance Achieved

Tests were run with the following conditions:

- 8-core Intel i7 machine as client
- Version 2.1 of panda-server installed on PandA
- PandA and client machine connected to same Gigabit ethernet switch
- 60 byte sample payload
- Using the commandline pandablocks hdf utility to write data to an SSD

When panda-webcontrol is installed with a single browser connected, the following results
were achieved:

- 50MBytes/s throughput
- PandA CPU usage about 75% (of both cores)
- local client CPU usage about 55% (of a single core)

When panda-webcontrol was not installed, the following results were achieved:

- 60MBytes/s throughput
- PandA CPU usage about 65% (of both cores)
- local client CPU usage about 60% (of a single core)

Increasing above these throughputs failed most scans with `DATA_OVERRUN`.

## Data overruns

If there is a `DATA_OVERRUN`, the server will stop sending data. The most recently
received `FrameData` from either `AsyncioClient.data` or `BlockingClient.data` may
be corrupt. This is the case if the `scaled` argument is set to False. The mechanism
the server uses to send raw unscaled data is only able to detect the corrupt frame after
it has already been sent. Conversely, the mechanism used to send scaled data aborts prior
to sending a corrupt frame.

The `write_hdf_files` function uses `scaled=False`, so your HDF file may include some
corrupt data in the event of an overrun.