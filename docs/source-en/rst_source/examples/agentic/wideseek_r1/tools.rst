.. _wideseek-r1-tools:

Tool Setup
==========

WideSeek-R1 provides two search backends:

- ``online`` mode for web-based search and webpage access.
- ``offline`` mode for retrieval against a Qdrant-based local knowledge base.

In the standard WideSeek-R1 workflow, offline tools are typically used for
training and QA-style evaluation, while online tools are used for WideSearch
evaluation. For the overall example flow, see :doc:`index`.

.. _wideseek-r1-online-tools:

Online Mode
-----------

Online mode uses `Serper <https://serper.dev>`__ for web search and
`Jina <https://jina.ai>`__ for webpage access.

API Keys
~~~~~~~~

Set the required API keys before running training or evaluation:

.. code-block:: bash

   export SERPER_API_KEY=your_serper_api_key
   export JINA_API_KEY=your_jina_api_key

Configuration
~~~~~~~~~~~~~

In your training or evaluation YAML, configure the tools section as follows:

.. code-block:: yaml

   tools:
     online: True
     use_jina: True
     enable_cache: True
     cache_file: "./webpage_cache.json"

The base WideSeek-R1 configurations are defined in
``examples/wideseek_r1/config/base_train.yaml`` and
``examples/wideseek_r1/config/base_eval.yaml``.

.. _wideseek-r1-offline-tools:

Offline Mode
------------

Offline mode uses a Qdrant-based retrieval service together with a local
corpus and webpage store.

Prerequisites
~~~~~~~~~~~~~

After completing the main environment setup in the
:doc:`installation guide <../../../start/installation>`, install the Qdrant
client:

.. code-block:: bash

   uv pip install qdrant-client==1.16.2

Data and Retriever Model
~~~~~~~~~~~~~~~~~~~~~~~~

Prepare the following assets:

- The local WideSeek-R1 corpus from
  `WideSeek-R1-Corpus <https://huggingface.co/datasets/RLinf/WideSeek-R1-Corpus>`__.
- The retriever model
  `intfloat/e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__.

At minimum, you will need:

- ``wiki_webpages.jsonl`` for webpage access.
- A Qdrant collection built for the retrieval corpus.
- A local path to the E5 retriever model.

Launch the Retrieval Service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit the variables at the top of
``examples/wideseek_r1/search_engine/launch_qdrant.sh`` so they match your
environment, especially:

- ``pages_file``
- ``retriever_path``
- ``qdrant_url``
- ``qdrant_collection_name``

Then start the retrieval service:

.. code-block:: bash

   bash examples/wideseek_r1/search_engine/launch_qdrant.sh

The service listens on port ``8000`` by default and exposes two endpoints:

- ``POST /retrieve`` for vector retrieval.
- ``POST /access`` for webpage content lookup.

Configuration
~~~~~~~~~~~~~

In your training or evaluation YAML, configure offline tools as follows:

.. code-block:: yaml

   tools:
     online: False
     search:
       server_addr: "127.0.0.1:8000"
       topk: 3

.. _wideseek-r1-tool-test:

Test the Tools
--------------

You can test the WideSeek-R1 tool worker directly.

Online mode:

.. code-block:: bash

   python rlinf/agents/wideseek_r1/tools.py --is_online true

Offline mode:

.. code-block:: bash

   python rlinf/agents/wideseek_r1/tools.py --is_online false

The online test requires ``SERPER_API_KEY`` and ``JINA_API_KEY``. The offline
test requires the local retrieval service to be available at the configured
``server_addr``.

See Also
--------

- :doc:`WideSeek-R1 Example <index>`
- :ref:`wideseek-r1-online-tools`
- :ref:`wideseek-r1-offline-tools`
- :ref:`wideseek-r1-tool-test`
