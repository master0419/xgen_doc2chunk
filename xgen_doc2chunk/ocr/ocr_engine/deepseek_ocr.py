# xgen_doc2chunk/ocr/ocr_engine/deepseek_ocr.py
# OCR class for DeepSeek-OCR / DeepSeek-OCR-2 models served via vLLM
"""
DeepSeek-OCR OCR engine.

DeepSeek-OCR / DeepSeek-OCR-2 is a vision model specialized for OCR tasks.
Unlike general Vision-Language models (Qwen2-VL, LLaVA, etc.) that accept
arbitrary instructions, DeepSeek-OCR's chat template is trained on a specific
format and the generic VllmOCR path produces empty / invalid responses.

This class enforces the DeepSeek-OCR vLLM recipe:
  - prompt: "Free OCR." (or other DeepSeek-specific instructions)
  - content order: image_url FIRST, then text (chat template requirement)
  - response parsing: tolerant of both str and list[dict] content
    (skip_special_tokens=False can produce list-shaped responses on some
    langchain-openai versions)

Reference:
  https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR-2.html

Server-side requirements (NOT enforced by this class — configure on the vLLM
server side):
  vllm serve deepseek-ai/DeepSeek-OCR-2 \\
    --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \\
    --no-enable-prefix-caching \\
    --mm-processor-cache-gb 0

Client-side extra_body should include (typically configured at llm_client
creation time, NOT here):
  - skip_special_tokens: False
  - vllm_xargs: {"ngram_size": 30, "window_size": 90, "whitelist_token_ids": [128821, 128822]}
"""
import logging
import os
from typing import Any, Optional

from xgen_doc2chunk.ocr.base import BaseOCR

logger = logging.getLogger("ocr-deepseek")

# Default model identifier — matches vLLM recipe.
DEFAULT_DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-OCR-2"


class DeepSeekOCR(BaseOCR):
    """
    OCR engine for DeepSeek-OCR / DeepSeek-OCR-2 served via vLLM
    OpenAI-compatible endpoint.

    Differs from :class:`VllmOCR` in three critical ways:
      1. ``DEFAULT_PROMPT`` is ``"Free OCR."`` (DeepSeek-OCR recipe).
      2. ``build_message_content`` puts ``image_url`` BEFORE ``text``
         (DeepSeek-OCR chat template requires image-first).
      3. ``convert_image_to_text`` uses tolerant response parsing
         (handles both string and list-shaped AIMessage.content) and treats
         empty responses as errors so the caller can preserve the original
         image tag instead of inserting an empty ``[Figure:]`` placeholder.

    Examples:
        >>> # Direct construction (lazy ChatOpenAI creation)
        >>> ocr = DeepSeekOCR(
        ...     base_url="http://vllm-server:8000/v1",
        ...     model="deepseek-ai/DeepSeek-OCR-2",
        ...     max_tokens=8192,
        ... )

        >>> # With pre-configured client (recommended when callers manage
        >>> # extra_body / vllm_xargs / skip_special_tokens themselves)
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(
        ...     model="deepseek-ai/DeepSeek-OCR-2",
        ...     base_url="http://vllm-server:8000/v1",
        ...     api_key="EMPTY",
        ...     temperature=0.0,
        ...     max_tokens=8192,
        ...     extra_body={
        ...         "skip_special_tokens": False,
        ...         "vllm_xargs": {
        ...             "ngram_size": 30,
        ...             "window_size": 90,
        ...             "whitelist_token_ids": [128821, 128822],
        ...         },
        ...     },
        ... )
        >>> ocr = DeepSeekOCR(llm_client=llm)
        >>> result = ocr.convert_image_to_text("/path/to/image.png")
    """

    # vLLM DeepSeek-OCR-2 recipe default prompt.
    # Other recipe-supported prompts:
    #   - "<image>\\n<|grounding|>Convert the document to markdown."
    #   - "Parse the figure."
    DEFAULT_PROMPT = "Free OCR."

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = "EMPTY",
        model: str = DEFAULT_DEEPSEEK_MODEL,
        llm_client: Optional[Any] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 8192,
    ):
        """
        Initialize DeepSeek-OCR engine.

        Args:
            base_url: vLLM server URL (e.g., "http://localhost:8000/v1").
                Ignored when ``llm_client`` is provided.
            api_key: API key. Default ``"EMPTY"`` (vLLM convention).
            model: Model identifier registered on the vLLM server.
                Default: ``"deepseek-ai/DeepSeek-OCR-2"``.
            llm_client: Pre-configured LangChain ``ChatOpenAI`` instance.
                When provided, ``base_url`` / ``api_key`` / ``temperature`` /
                ``max_tokens`` are ignored — the caller is responsible for
                ``extra_body`` (``skip_special_tokens`` / ``vllm_xargs``) too.
            prompt: OCR instruction. ``None`` or empty → ``DEFAULT_PROMPT``.
            temperature: Sampling temperature. DeepSeek-OCR recipe uses 0.0.
            max_tokens: Max completion tokens. DeepSeek-OCR recipe uses 2048,
                8192 is safer for long documents. Only used when ``llm_client``
                is None.
        """
        cleaned_prompt = (prompt or "").strip()
        if not cleaned_prompt:
            cleaned_prompt = self.DEFAULT_PROMPT

        if llm_client is None:
            if base_url is None:
                raise ValueError("Either base_url or llm_client is required.")

            from langchain_openai import ChatOpenAI  # noqa: PLC0415

            client_kwargs: dict[str, Any] = {
                "model": model,
                "base_url": base_url,
                "api_key": api_key,
                "temperature": temperature,
            }
            if max_tokens is not None:
                client_kwargs["max_tokens"] = max_tokens

            llm_client = ChatOpenAI(**client_kwargs)
            logger.info(
                f"[DeepSeek-OCR] Client created: base_url={base_url}, model={model}"
            )

        super().__init__(llm_client=llm_client, prompt=cleaned_prompt)
        self.model = model
        self.base_url = base_url
        logger.info("[DeepSeek-OCR] Initialization completed")

    @property
    def provider(self) -> str:
        return "deepseek-ocr"

    # ---- Message construction (DeepSeek-OCR specific) ----

    def build_message_content(self, b64_image: str, mime_type: str) -> list:
        """
        Build user content for DeepSeek-OCR chat completions.

        Per vLLM DeepSeek-OCR-2 recipe, the image_url block MUST precede the
        text block. If text comes first, the chat template generates a prompt
        outside the model's training distribution and the model returns an
        empty response (or HTTP 400).

        Returns:
            Content list ``[image_url, text]`` (image-first order is required).
        """
        return [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
            },
            {"type": "text", "text": self.prompt},
        ]

    # ---- Response parsing (tolerant) ----

    @staticmethod
    def _extract_text_safe(content: Any) -> str:
        """
        Extract plain text from ``AIMessage.content``.

        LangChain typically returns a string for OpenAI-compatible chat
        completions, but with ``skip_special_tokens=False`` some
        langchain-openai versions wrap multimodal responses as
        ``list[dict | str]``. ``str.strip`` raises ``AttributeError`` on
        lists, which the base class would convert to ``"[Image conversion
        error: 'list' object has no attribute 'strip']"`` — visible but
        unhelpful. This method handles both shapes.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text_part = item.get("text")
                    if isinstance(text_part, str):
                        parts.append(text_part)
            return "".join(parts)
        return ""

    # ---- Sync invocation (overrides BaseOCR for tolerant parsing) ----

    def convert_image_to_text(self, image_path: str) -> Optional[str]:
        """
        Convert image to text via DeepSeek-OCR.

        Overrides :meth:`BaseOCR.convert_image_to_text` to:
          1. Use :meth:`_extract_text_safe` instead of ``response.content.strip()``.
          2. Treat empty responses as errors (returns
             ``"[Image conversion error: ...]"``) so callers can keep the
             original ``[Image:...]`` tag intact instead of inserting an
             empty ``[Figure:]`` placeholder.

        Args:
            image_path: Local image file path.

        Returns:
            On success: ``"[Figure:{ocr_text}]"``.
            On failure: ``"[Image conversion error: {reason}]"`` (caller
            should detect this prefix and preserve the original tag).
        """
        from xgen_doc2chunk.ocr.ocr_processor import (  # noqa: PLC0415
            _b64_from_file,
            _get_mime_type,
        )
        from langchain_core.messages import HumanMessage  # noqa: PLC0415

        try:
            b64_image = _b64_from_file(image_path)
            mime_type = _get_mime_type(image_path)

            content = self.build_message_content(b64_image, mime_type)
            message = HumanMessage(content=content)

            logger.debug(
                "[DeepSeek-OCR] invoke: image=%s, prompt=%r",
                os.path.basename(image_path), self.prompt,
            )

            response = self.llm_client.invoke([message])
            text = self._extract_text_safe(response.content).strip()

            if not text:
                # vLLM server flag missing (--logits_processors / --no-enable-prefix-caching
                # / --mm-processor-cache-gb 0) or chat template mismatch.
                # Raise so the except branch returns a recognizable error string;
                # the caller's "[Image conversion error:" prefix check then keeps
                # the original tag rather than inserting an empty [Figure:].
                raise ValueError(
                    "DeepSeek-OCR returned empty response "
                    "(check vLLM server flags and model identifier)"
                )

            result = f"[Figure:{text}]"
            logger.info(
                f"[DeepSeek-OCR] Image to text conversion completed: "
                f"{os.path.basename(image_path)}"
            )
            return result

        except Exception as e:
            logger.error(
                f"[DeepSeek-OCR] Image to text conversion failed: "
                f"{image_path}, error: {e}"
            )
            return f"[Image conversion error: {str(e)}]"
