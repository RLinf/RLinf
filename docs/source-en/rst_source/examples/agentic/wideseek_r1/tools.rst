.. _wideseek-r1-tools:

Tool Setup
==========

WideSeek-R1 provides two search backends:

- ``online`` mode for live web search and webpage access.
- ``offline`` mode for retrieval against a local Qdrant-based knowledge base.

In the standard workflow, offline tools are typically used for training and
standard QA evaluation, while online tools are used for WideSearch evaluation.

.. contents::
   :depth: 2
   :local:

.. _wideseek-r1-online-tools:

Online Mode
-----------

Online mode uses `Serper <https://serper.dev>`__ for web search and
`Jina AI <https://jina.ai>`__ for webpage access.

API Keys
~~~~~~~~

Export the required API keys before running training or evaluation:

.. code-block:: bash

   export SERPER_API_KEY=your_serper_api_key
   export JINA_API_KEY=your_jina_api_key

Configuration
~~~~~~~~~~~~~

In the YAML config under ``examples/agent/wideseek_r1/config``, set:

.. code-block:: yaml

   tools:
     online: True
     use_jina: True
     enable_cache: True
     cache_file: "./webpage_cache.json"

.. _wideseek-r1-offline-tools:

Offline Mode
------------

Offline mode uses a local Qdrant retrieval service together with a local corpus
and webpage store.

Prerequisites
~~~~~~~~~~~~~

After completing the base setup from the
:doc:`installation guide <../../../start/installation>`, install the Qdrant
client:

.. code-block:: bash

   uv pip install qdrant-client==1.16.2

Download the Corpus and Retriever
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prepare the following assets:

- `Wiki-2018-Corpus <https://huggingface.co/datasets/RLinf/Wiki-2018-Corpus>`__
- `intfloat/e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__

The corpus package includes:

- ``wiki_corpus.jsonl`` for retrieval snippets.
- ``wiki_webpages.jsonl`` for webpage content lookup.
- ``qdrant/`` containing the Qdrant collection files.

Launch the Retrieval Service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Start Qdrant in the corpus directory:

   .. code-block:: bash

      cd /PATH/TO/Wiki-2018-Corpus/qdrant
      ./qdrant

   This process must stay alive. Running it inside ``tmux`` is recommended.

2. Get the host IP address for the Qdrant service:

   .. code-block:: bash

      hostname -I

3. Edit
   `examples/agent/wideseek_r1/search_engine/launch_qdrant.sh`
   and update these variables:

   - ``pages_file``: ``/PATH/TO/Wiki-2018-Corpus/wiki_webpages.jsonl``
   - ``retriever_path``: ``/PATH/TO/e5-model``
   - ``qdrant_url``: for example ``http://<host_ip>:6333``

4. Start the retrieval service:

   .. code-block:: bash

      bash examples/agent/wideseek_r1/search_engine/launch_qdrant.sh

We recommend running this retrieval service on the same machine as training or
evaluation to avoid unnecessary network latency. If you run it elsewhere,
configure ``tools.search.server_addr`` accordingly. The default address is
``localhost:8000``.

The retrieval service listens on port ``8000`` by default and exposes:

- ``POST /retrieve`` for vector retrieval.
- ``POST /access`` for webpage content lookup.

Because Qdrant retrieval runs on CPU, only the E5 retriever model consumes GPU
memory after the service starts.

Configuration
~~~~~~~~~~~~~

In your YAML config, set:

.. code-block:: yaml

   tools:
     online: False

If the retrieval service is not running on the local machine, also set:

.. code-block:: yaml

   tools:
     search:
       server_addr: "HOST:8000"

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

The online test requires ``SERPER_API_KEY`` and ``JINA_API_KEY``.

The offline test requires the local retrieval service to be reachable at the
configured ``server_addr``.
