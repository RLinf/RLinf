使用 ``InferenceHTTPClient`` 调用 SGLang
=========================================

:class:`rlinf.utils.http_client.InferenceHTTPClient` 是对 ``requests`` 与
``aiohttp`` 的轻薄封装，面向单个 sglang **router 或 server** 的 base URL。
同步和异步接口都有；异步接口共享一个懒加载的 ``aiohttp.ClientSession``，请
使用 ``async with``（或主动调用 ``aclose()``）来释放 socket。

本文假定你已经有了 ``router_url`` ——如何启动一个，参见
:doc:`SGLang Server 与 Router <sglang_server>`。

接口一览
--------

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - 方法
     - Endpoint
     - 说明
   * - ``generate`` / ``async_generate``
     - ``POST /generate``
     - sglang 原生接口；传入 ``prompt`` *或* ``input_ids``，外加 ``sampling_params``。
   * - ``chat_completion`` / ``async_chat_completion``
     - ``POST /v1/chat/completions``
     - 兼容 OpenAI；传入 ``messages`` 和 ``model``（多余 kwargs 会作为 JSON 字段透传）。
   * - ``health`` / ``async_health``
     - ``GET /health``
     - 返回 ``bool``；从不抛出。

同步——适合短脚本与测试
-----------------------

.. code-block:: python

   from rlinf.utils.http_client import InferenceHTTPClient

   client = InferenceHTTPClient(router_url)         # 此时未建 session——同步方法用 requests
   assert client.health()

   out = client.generate(
       prompt="The capital of France is",
       sampling_params={"temperature": 0.0, "max_new_tokens": 32},
   )

   reply = client.chat_completion(
       messages=[{"role": "user", "content": "Hello"}],
       model=cfg.router_server_args.model_path,     # router 接受任意字符串
       temperature=0.0,
       max_tokens=32,
   )

异步——并发发送多请求
---------------------

使用 ``async with`` 让 aiohttp session 在退出时自动关闭：

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

OpenAI 风格 chat 同理：

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

并发与超时
----------

- ``connect_timeout``（默认 ``10.0`` 秒）只约束 TCP 连接阶段；**读** 端故意
  不设上限——生成请求可能耗时很久。如需请求级别的截止时间，请在外层用
  ``asyncio.wait_for`` 包裹。
- ``max_connections``（默认 ``16384``）约束 aiohttp ``TCPConnector``。超过上限
  的 task 不会失败——aiohttp 会在 connector 内排队——但并发度不会超过这个上限。
  RL 风格的大扇出场景，记得把 ``ulimit -n`` 调到足够大。
- 如果要更细粒度地控制 *逻辑* 并发（例如每个 rollout 样本一个许可），在调用方
  用 ``asyncio.Semaphore`` 包裹即可。``max_connections`` 只是 socket 数量的硬上限。

直接连后端 server
-----------------

``InferenceHTTPClient`` 接受任意 base URL，因此也可绕过 router 直连单个 server：

.. code-block:: python

   from rlinf.utils.http_client import InferenceHTTPClient

   server_urls = server_group.get_server_url().wait()
   direct = InferenceHTTPClient(server_urls[0])

这在排查问题时很有用——直连引擎可以确认问题是出在 router 路由还是后端本身。
