Calling SGLang with ``InferenceHTTPClient``
===========================================

:class:`rlinf.utils.http_client.InferenceHTTPClient` is a thin wrapper around
``requests`` and ``aiohttp`` that targets a single sglang **router or server**
base URL. Both sync and async methods are provided; async methods share one
lazily-created ``aiohttp.ClientSession``, so use ``async with`` (or call
``aclose()``) to release sockets.

This guide assumes you already have a ``router_url`` — see
:doc:`SGLang Server & Router <sglang_server>` for how to launch one.

Endpoints
---------

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Method
     - Endpoint
     - Notes
   * - ``generate`` / ``async_generate``
     - ``POST /generate``
     - sglang-native; pass ``prompt`` *or* ``input_ids``, plus ``sampling_params``.
   * - ``chat_completion`` / ``async_chat_completion``
     - ``POST /v1/chat/completions``
     - OpenAI-compatible; pass ``messages`` and ``model`` (extra kwargs forwarded as JSON).
   * - ``health`` / ``async_health``
     - ``GET /health``
     - Returns ``bool``; never raises.

Sync — Short Scripts and Tests
------------------------------

.. code-block:: python

   from rlinf.utils.http_client import InferenceHTTPClient

   client = InferenceHTTPClient(router_url)         # no session yet — sync methods use `requests`
   assert client.health()

   out = client.generate(
       prompt="The capital of France is",
       sampling_params={"temperature": 0.0, "max_new_tokens": 32},
   )

   reply = client.chat_completion(
       messages=[{"role": "user", "content": "Hello"}],
       model=cfg.router_server_args.model_path,     # any string the router accepts
       temperature=0.0,
       max_tokens=32,
   )

Async — Fan Out Many Requests in Parallel
-----------------------------------------

Use ``async with`` so the aiohttp session is closed on exit:

.. code-block:: python

   import asyncio

   from rlinf.utils.http_client import InferenceHTTPClient


   async def fan_out(router_url: str, prompts: list[str]) -> list[dict]:
       async with InferenceHTTPClient(router_url) as client:
           tasks = [
               client.async_generate(
                   prompt=p,
                   sampling_params={"temperature": 0.0, "max_new_tokens": 32},
               )
               for p in prompts
           ]
           return await asyncio.gather(*tasks)


   results = asyncio.run(fan_out(router_url, prompts))

For OpenAI-style chat:

.. code-block:: python

   import asyncio

   from rlinf.utils.http_client import InferenceHTTPClient


   async def chat_many(
       router_url: str,
       model: str,
       messages_list: list[list[dict]],
   ) -> list[dict]:
       async with InferenceHTTPClient(router_url) as client:
           tasks = [
               client.async_chat_completion(
                   messages=m, model=model, temperature=0.0, max_tokens=32
               )
               for m in messages_list
           ]
           return await asyncio.gather(*tasks)

Concurrency and Timeouts
------------------------

- ``connect_timeout`` (default ``10.0`` s) bounds the TCP connect phase only;
  the **read** side is uncapped on purpose — generation can take arbitrarily
  long. Wrap calls in your own ``asyncio.wait_for`` if you need a request-level
  deadline.
- ``max_connections`` (default ``16384``) caps the aiohttp ``TCPConnector``.
  Tasks scheduled above the cap don't fail — aiohttp queues them inside the
  connector — but you won't see more parallelism than the limit allows. For
  RL-style fan-out, make sure ``ulimit -n`` covers that many sockets.
- For tighter control of *logical* concurrency (e.g. one permit per rollout
  sample), wrap calls in an ``asyncio.Semaphore`` on the caller side.
  ``max_connections`` is only a hard ceiling on socket count.

Pointing the Client Elsewhere
-----------------------------

``InferenceHTTPClient`` takes any base URL, so it also works against a single
backend (skip the router and grab a server URL directly):

.. code-block:: python

   from rlinf.utils.http_client import InferenceHTTPClient

   server_urls = server_group.get_server_url().wait()
   direct = InferenceHTTPClient(server_urls[0])

This is useful for debugging an individual engine — bypassing the router lets
you confirm whether a problem is in routing or in the backend.
