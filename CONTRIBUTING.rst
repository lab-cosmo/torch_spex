
Running Benchmarks
------------------


To run the benchmark with the currently installed itorch_spex version using default asv timinig paramaters use 

  .. code-block:: bash

      tox -e benchmarks


See https://asv.readthedocs.io/en/stable/writing_benchmarks.html#timing
and https://asv.readthedocs.io/en/stable/benchmarks.html#timing-benchmarks
for explanation of benchmark attributes rounds, repeat, and number.
They can be manipulated using


  .. code-block:: bash

      tox -e benchmarks -- -a rounds=4 -a repeat=2,10,5  -a number=20

To run the benchmark quickly to check if they run to the end, you can use

  .. code-block:: bash

      tox -e benchmarks -- --quick

To run the commit using ``benchmarks-commit`` and specify the commit id

  .. code-block:: bash

      tox -e benchmarks-commit -- 'HEAD^!' # runs HEAD
      tox -e benchmarks-commit -- 08cfe0d2 # runs commit 08cfe0d2

Here again the commands to control the timing parameters can be added before the commit id

  .. code-block:: bash

      tox -e benchmarks-commit -- --quick 08cfe0d2
