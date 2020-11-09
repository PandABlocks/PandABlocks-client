.. include:: ../README.rst

How the documentation is structured
-----------------------------------

.. rst-class:: columns

:ref:`tutorials`
~~~~~~~~~~~~~~~~

Tutorials for installation, library and commandline usage. New users start here.

.. rst-class:: columns

:ref:`how-to`
~~~~~~~~~~~~~

Practical step-by-step guides for the more experienced user.

.. rst-class:: columns

:ref:`explanations`
~~~~~~~~~~~~~~~~~~~

Explanation of how the library works and why it works that way.

.. rst-class:: columns

:ref:`reference`
~~~~~~~~~~~~~~~~

Technical reference material, for classes, methods, APIs, commands, and contributing to the project.

.. rst-class:: endcolumns

About the documentation
~~~~~~~~~~~~~~~~~~~~~~~

`Why is the documentation structured this way? <https://documentation.divio.com>`_

.. toctree::
    :caption: Tutorials
    :name: tutorials

    tutorials/installation
    tutorials/load-save
    tutorials/control
    tutorials/commandline-hdf

.. toctree::
    :caption: How-to Guides
    :name: how-to

    how-to/library-hdf
    how-to/poll-changes

.. toctree::
    :caption: Explanations
    :name: explanations

    explanations/sans-io
    explanations/performance

.. rst-class:: no-margin-after-ul

.. toctree::
    :caption: Reference
    :name: reference

    reference/api
    reference/changelog
    reference/contributing

* :ref:`genindex`
