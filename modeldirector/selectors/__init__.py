"""Selector backends for ModelDirector.

Each module in this package implements a ``score(prompt, models) -> dict[str, ModelScore]``
function that the engine can call to obtain per-model confidence scores for a user
prompt. Two backends ship today:

  * ``llm``  - default; asks a configurable LiteLLM-routed LLM (GPT-5 Mini,
              Claude Haiku, etc.) to score each candidate. Slow, costs money,
              gives per-axis scores (reasoning/coding/context/creativity).
  * ``laya`` - the Laya decision engine: a non-autoregressive System-1
              model that scores every candidate in a single forward pass
              (~33 ms on GPU, ~free on CPU after weights download). Returns
              a single probability per candidate, which we map to a single
              ``overall`` 0-100 score.

Select the backend with ``selector.backend: llm|laya`` in the YAML config
or with the ``--selector-backend`` CLI flag.
"""