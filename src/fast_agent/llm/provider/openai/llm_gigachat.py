from __future__ import annotations

import json
import ssl
import time
import uuid
from types import SimpleNamespace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type

import httpx
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    ContentBlock,
    TextContent,
)
from openai import AsyncOpenAI, AuthenticationError
from pydantic_core import from_json

from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt import Prompt
from fast_agent.event_progress import ProgressAction
from fast_agent.llm.provider.openai.llm_openai_compatible import OpenAICompatibleLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import TurnUsage
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.types import LlmStopReason, PromptMessageExtended, RequestParams

if TYPE_CHECKING:
    from mcp import Tool

logger = get_logger(__name__)

# Default URLs for different authentication methods
GIGACHAT_API_KEY_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
GIGACHAT_CERT_BASE_URL = "https://gigachat-ift.sberdevices.delta.sbrf.ru/v1/"
GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
DEFAULT_GIGACHAT_MODEL = "GigaChat"
EVENT_STREAM_HEADER = "text/event-stream"


class GigaChatLLM(OpenAICompatibleLLM):
    """
    GigaChat LLM provider supporting two authentication methods:
    1. API key authentication (public API) - requires OAuth token exchange via OpenAI-compatible client
    2. Certificate-based authentication (IFT endpoint) - uses direct HTTPX streaming (legacy, proven path)
    """

    # Fields to exclude when building provider-specific arguments (for legacy HTTP path)
    GIGACHAT_EXCLUDE_FIELDS = {
        # Base FastAgentLLM fields
        "messages",
        "model",
        "maxTokens",
        "systemPrompt",
        "parallel_tool_calls",
        "use_history",
        "max_iterations",
        "template_vars",
        "mcp_metadata",
        "stopSequences",
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, provider=Provider.GIGACHAT, **kwargs)
        # Cache for OAuth access token (for API key authentication)
        self._access_token_cache: dict[str, tuple[str, float]] = {}

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages: list[PromptMessageExtended],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> tuple[ModelT | None, PromptMessageExtended]:
        """
        Override to prevent mutation of original messages.

        IMPORTANT: Create a copy of the last message before adding structured output
        instructions to prevent contamination when messages are passed to child agents
        in router workflows. This follows the same pattern as BedrockLLM.
        """
        from fast_agent.interfaces import ModelT

        # Create a copy of the last message to avoid mutating the original
        # This is critical for router agents that pass the same messages to child agents
        try:
            copied_messages = multipart_messages[:-1] + [
                multipart_messages[-1].model_copy(deep=True)
            ]
        except Exception:
            # Fallback: construct a minimal copy if model_copy is unavailable
            copied_messages = list(multipart_messages[:-1])
            last_msg = multipart_messages[-1]
            copied_messages.append(
                PromptMessageExtended(role=last_msg.role, content=list(last_msg.content))
            )

        # Call parent method with copied messages (it will add instructions to the copy)
        return await super()._apply_prompt_provider_specific_structured(
            copied_messages, model, request_params
        )

    # ---------------------------------------------------------------------
    # Shared configuration helpers
    # ---------------------------------------------------------------------

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize GigaChat-specific default parameters"""
        base_params = super()._initialize_default_params(kwargs)

        # Get model from config or use default
        config_model = None
        if self.context.config and self.context.config.gigachat:
            config_model = getattr(self.context.config.gigachat, "model", None)

        chosen_model = kwargs.get("model", config_model or DEFAULT_GIGACHAT_MODEL)
        base_params.model = chosen_model

        return base_params

    def _get_gigachat_config(self):
        """Get GigaChat configuration section"""
        if not self.context or not self.context.config:
            return None
        return getattr(self.context.config, "gigachat", None)

    # ---------------------------------------------------------------------
    # Auth method selection
    # ---------------------------------------------------------------------

    def _determine_auth_method(
        self,
    ) -> tuple[str, bool, ssl.SSLContext | None, tuple[str, str] | str | None, str | None]:
        """
        Determine authentication method and return (base_url, verify_ssl, ssl_context, cert_tuple, ca_bundle).

        Returns:
            Tuple of (base_url, verify_ssl_certs, ssl_context, cert_tuple, ca_bundle)
            - cert_tuple: (cert_path, key_path) tuple or single cert_path string for httpx.AsyncClient cert parameter
            - ca_bundle: Path to CA bundle for server certificate verification
        """
        config = self._get_gigachat_config()

        # Get explicit values from config
        explicit_base_url = getattr(config, "base_url", None) if config else None
        explicit_verify_ssl = getattr(config, "verify_ssl_certs", None) if config else None
        api_key = getattr(config, "api_key", None) if config else None

        # Check for certificate configuration (new and legacy formats)
        certificate_path = getattr(config, "certificate_path", None) if config else None
        cert_path = getattr(config, "cert_path", None) if config else None
        key_path = getattr(config, "key_path", None) if config else None
        ca_bundle = getattr(config, "ca_bundle", None) if config else None

        # Also check environment variable for API key
        if not api_key:
            import os

            api_key = os.getenv("GIGACHAT_API_KEY")

        # Determine authentication method
        # Priority: certificate_path (new) > cert_path/key_path (legacy) > api_key
        if certificate_path or (cert_path and key_path) or cert_path:
            # Certificate-based authentication
            cert_tuple: tuple[str, str] | str | None = None
            ca_bundle_path: str | None = None
            ssl_context: ssl.SSLContext | None = None

            # Handle legacy format (cert_path + key_path)
            if cert_path and key_path:
                cert_path_obj = Path(cert_path).expanduser()
                key_path_obj = Path(key_path).expanduser()

                if not cert_path_obj.exists():
                    raise ProviderKeyError(
                        "GigaChat certificate not found",
                        f"The certificate file at '{cert_path}' does not exist.\n"
                        "Please check the cert_path in your configuration.",
                    )
                if not key_path_obj.exists():
                    raise ProviderKeyError(
                        "GigaChat key not found",
                        f"The key file at '{key_path}' does not exist.\n"
                        "Please check the key_path in your configuration.",
                    )

                cert_tuple = (str(cert_path_obj), str(key_path_obj))

                # Use ca_bundle if provided, otherwise use cert_path for server verification
                if ca_bundle:
                    ca_bundle_path_obj = Path(ca_bundle).expanduser()
                    if not ca_bundle_path_obj.exists():
                        raise ProviderKeyError(
                            "GigaChat CA bundle not found",
                            f"The CA bundle file at '{ca_bundle}' does not exist.\n"
                            "Please check the ca_bundle in your configuration.",
                        )
                    ca_bundle_path = str(ca_bundle_path_obj)
                    ssl_context = ssl.create_default_context(cafile=ca_bundle_path)
                else:
                    # Use cert_path for server verification if no ca_bundle
                    ssl_context = ssl.create_default_context(cafile=str(cert_path_obj))

                logger.debug(
                    "Using certificate-based authentication (legacy format) "
                    f"with cert: {cert_path}, key: {key_path}"
                )

            # Handle new format (certificate_path - single file)
            elif certificate_path:
                cert_path_obj = Path(certificate_path).expanduser()
                if not cert_path_obj.exists():
                    raise ProviderKeyError(
                        "GigaChat certificate not found",
                        f"The certificate file at '{certificate_path}' does not exist.\n"
                        "Please check the certificate_path in your configuration.",
                    )

                cert_tuple = str(cert_path_obj)

                # Create SSL context with certificate for server verification
                ssl_context = ssl.create_default_context(cafile=str(cert_path_obj))

                logger.debug(
                    "Using certificate-based authentication (new format) "
                    f"with cert: {certificate_path}"
                )

            # Use explicit base_url or default for certificate auth
            base_url = explicit_base_url or GIGACHAT_CERT_BASE_URL
            verify_ssl = explicit_verify_ssl if explicit_verify_ssl is not None else True

            return base_url, verify_ssl, ssl_context, cert_tuple, ca_bundle_path

        elif api_key:
            # API key authentication
            # Use explicit base_url or default for API key auth
            base_url = explicit_base_url or GIGACHAT_API_KEY_BASE_URL
            verify_ssl = explicit_verify_ssl if explicit_verify_ssl is not None else False

            logger.debug("Using API key authentication")
            return base_url, verify_ssl, None, None, None

        else:
            # No authentication method specified
            raise ProviderKeyError(
                "GigaChat authentication not configured",
                "Either 'api_key', 'certificate_path', or 'cert_path'/'key_path' must be provided in the "
                "gigachat configuration.\nPlease add one of these to your fastagent.secrets.yaml file.",
            )

    def _base_url(self) -> str:
        """Get base URL based on authentication method"""
        base_url, _, _, _, _ = self._determine_auth_method()
        return base_url

    def _api_key(self) -> str:
        """Get API key (credentials) for authentication"""
        config = self._get_gigachat_config()
        if config and hasattr(config, "api_key") and config.api_key:
            return config.api_key

        # Fall back to environment variable
        import os

        api_key = os.getenv("GIGACHAT_API_KEY")
        if api_key:
            return api_key

        # If using certificate auth, API key is not required
        config = self._get_gigachat_config()
        if config:
            # Check for certificate configuration (new and legacy formats)
            if (hasattr(config, "certificate_path") and config.certificate_path) or (
                hasattr(config, "cert_path") and config.cert_path
            ):
                return ""  # Empty string for certificate-based auth

        raise ProviderKeyError(
            "GigaChat API key not configured",
            "The GigaChat API key is required for API key authentication.\n"
            "Add it to your configuration file under gigachat.api_key "
            "or set the GIGACHAT_API_KEY environment variable.",
        )

    # ---------------------------------------------------------------------
    # OAuth for API-key mode
    # ---------------------------------------------------------------------

    async def _get_access_token(self, credentials: str, scope: str = "GIGACHAT_API_PERS") -> str:
        """
        Get OAuth access token from GigaChat API using credentials (API-key mode).
        """
        # Check cache first
        cache_key = f"{credentials}:{scope}"
        if cache_key in self._access_token_cache:
            token, expires_at = self._access_token_cache[cache_key]
            # Check if token is still valid (with 60 second buffer)
            if time.time() < expires_at - 60:
                logger.debug("Using cached GigaChat access token")
                return token
            logger.debug("Cached GigaChat token expired, requesting new one")
            del self._access_token_cache[cache_key]

        # Request new token
        try:
            async with httpx.AsyncClient(verify=False) as client:
                headers = {
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    "Authorization": f"Basic {credentials}",
                    "RqUID": str(uuid.uuid4()),
                }
                data = {"scope": scope}

                # Important: our logger.debug signature is (message, *, name=None, context=None, **data)
                # so we must not pass scope as a positional argument (it would become the 'name' field
                # and break Event validation when it's not a string). Use f-string instead.
                logger.debug(f"Requesting GigaChat OAuth token with scope: {scope}")
                response = await client.post(
                    GIGACHAT_OAUTH_URL,
                    headers=headers,
                    data=data,
                    timeout=30.0,
                )
                response.raise_for_status()

                token_data = response.json()
                access_token = token_data.get("access_token")
                expires_in = token_data.get("expires_in", 1800)
                expires_at = time.time() + expires_in

                if not access_token:
                    raise ProviderKeyError(
                        "Invalid OAuth response",
                        "GigaChat OAuth endpoint did not return access_token.\n"
                        f"Response: {token_data}",
                    )

                # Cache the token
                self._access_token_cache[cache_key] = (access_token, float(expires_at))
                logger.debug(
                    f"Successfully obtained GigaChat access token, expires_at={expires_at}"
                )

                return access_token

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ProviderKeyError(
                    "Invalid GigaChat credentials",
                    "The GigaChat credentials were rejected during OAuth token request.\n"
                    "Please check that your api_key (credentials) is valid and not expired.",
                ) from e
            raise ProviderKeyError(
                "GigaChat OAuth request failed",
                f"Failed to obtain OAuth token: HTTP {e.response.status_code}\n"
                f"Response: {e.response.text}",
            ) from e
        except httpx.RequestError as e:
            raise ProviderKeyError(
                "GigaChat OAuth request failed",
                f"Network error while requesting OAuth token: {str(e)}",
            ) from e
        except Exception as e:  # pragma: no cover - defensive
            raise ProviderKeyError(
                "GigaChat OAuth request failed",
                f"Unexpected error while requesting OAuth token: {str(e)}",
            ) from e

    # ---------------------------------------------------------------------
    # OpenAI-compatible client (used only in API-key mode)
    # ---------------------------------------------------------------------

    def _openai_client(self) -> AsyncOpenAI:
        """
        Create an OpenAI client instance with GigaChat-specific configuration.

        - API key mode: use AsyncOpenAI with OAuth token
        - Certificate mode: bypassed (direct HTTPX path is used instead)
        """
        try:
            base_url, verify_ssl, ssl_context, cert_tuple, ca_bundle = self._determine_auth_method()

            # If we are in certificate mode, we do NOT want to use AsyncOpenAI client.
            # The direct HTTPX streaming path will be used instead.
            if ssl_context is not None or cert_tuple is not None:
                raise ProviderKeyError(
                    "GigaChat certificate mode not supported by OpenAI client",
                    "Certificate-based GigaChat connections use a dedicated HTTPX path instead of AsyncOpenAI.",
                )

            kwargs: dict[str, Any] = {
                "base_url": base_url,
            }

            # Get credentials for API key authentication
            credentials = self._api_key()
            if not credentials:
                raise ProviderKeyError(
                    "GigaChat credentials required",
                    "API key authentication requires credentials.",
                )

            # For API key auth, we need to handle OAuth token
            # Since OpenAI client is created synchronously but token is async,
            # we'll need to get token synchronously
            import asyncio

            config = self._get_gigachat_config()
            scope = getattr(config, "scope", "GIGACHAT_API_PERS") if config else "GIGACHAT_API_PERS"

            try:
                # If there's a running event loop, we need to handle it carefully
                asyncio.get_running_loop()

                import concurrent.futures

                def get_token_in_thread() -> str:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self._get_access_token(credentials, scope)
                        )
                    finally:
                        new_loop.close()

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(get_token_in_thread)
                    access_token = future.result(timeout=30)
            except RuntimeError:
                # No running event loop, safe to use asyncio.run
                access_token = asyncio.run(self._get_access_token(credentials, scope))

            kwargs["api_key"] = access_token

            # Raw HTTPX client for API-key mode (without client certs)
            http_client = httpx.AsyncClient(verify=verify_ssl)
            kwargs["http_client"] = http_client

            # Add custom headers if configured
            default_headers = self._default_headers()
            if default_headers:
                kwargs["default_headers"] = default_headers

            return AsyncOpenAI(**kwargs)

        except ProviderKeyError:
            raise
        except Exception as e:
            if isinstance(e, AuthenticationError):
                raise ProviderKeyError(
                    "Invalid GigaChat credentials",
                    "The configured GigaChat credentials were rejected.\n"
                    "Please check that your API key or certificate is valid.",
                ) from e
            raise ProviderKeyError(
                "Failed to create GigaChat client",
                f"Error creating GigaChat client: {str(e)}",
            ) from e

    # ---------------------------------------------------------------------
    # Legacy certificate-based HTTP path (direct GigaChat API)
    # ---------------------------------------------------------------------

    def _use_cert_auth(self) -> bool:
        """Return True if configuration indicates certificate-based authentication."""
        config = self._get_gigachat_config()
        if not config:
            return False

        api_key = getattr(config, "api_key", None)
        if api_key:
            return False

        if getattr(config, "certificate_path", None):
            return True

        cert_path = getattr(config, "cert_path", None)
        key_path = getattr(config, "key_path", None)
        if cert_path and key_path:
            return True

        return False

    def _resolve_ssl_verify(self) -> Any:
        """Resolve SSL verification settings for server certificate validation."""
        config = self._get_gigachat_config()
        if not config:
            return True

        # Prefer explicit ssl_context if present
        ssl_context = getattr(config, "ssl_context", None)
        if ssl_context:
            return ssl_context

        verify = getattr(config, "verify_ssl_certs", None)
        if verify is not None:
            return verify

        ca_bundle = getattr(config, "ca_bundle", None)
        if ca_bundle:
            return ca_bundle

        return True

    def _resolve_client_cert(self) -> tuple[str, str] | str | None:
        """Resolve client certificate parameters for httpx cert parameter."""
        config = self._get_gigachat_config()
        if not config:
            return None

        cert_path = getattr(config, "cert_path", None)
        key_path = getattr(config, "key_path", None)

        # If both cert and key paths are provided, return tuple
        if cert_path and key_path:
            return (str(Path(cert_path).expanduser()), str(Path(key_path).expanduser()))

        # If only cert_path is provided (PEM file with both cert and key)
        if cert_path and not key_path:
            return str(Path(cert_path).expanduser())

        return None

    async def _gigachat_completion_cert(
        self,
        message: Optional[List[Dict[str, Any]]],
        request_params: Optional[RequestParams] = None,
        tools: Optional[List[Tool]] = None,
    ) -> PromptMessageExtended:
        """Direct HTTPX completion path for certificate-based authentication (legacy, working on local machine)."""
        request_params = self.get_request_params(request_params=request_params)
        response_blocks: List[ContentBlock] = []

        model_name = self.default_request_params.model or DEFAULT_GIGACHAT_MODEL

        messages: List[Dict[str, Any]] = []
        system_prompt = self.instruction or request_params.systemPrompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # message parameter already contains ALL conversation history
        # (converted from multipart_messages in _apply_prompt_provider_specific)
        # No need to use self.history.get() - this follows the same pattern as base class
        if message is not None:
            messages.extend(message)
            self.logger.debug(
                f"Gigachat cert mode: sending {len(message)} messages to API",
                data={"message_count": len(message), "roles": [msg.get("role") for msg in message]},
            )

        # Проверить, что в запросе вообще есть user‑сообщение.
        # Не блокируем запрос из‑за пустого текста (это приводило к тому,
        # что первое сообщение «терялось»), но защищаемся от полностью
        # отсутствующего user‑роля.
        has_user_message = any(msg.get("role") == "user" for msg in messages)
        if not has_user_message:
            self.logger.error("No user message to send to GigaChat API")
            return self._stream_failure_response(
                ValueError("No user message - nothing to send to GigaChat API"), model_name
            )

        functions = self._prepare_functions_cert(tools)
        arguments = self._prepare_api_request_cert(messages, functions, request_params)

        self.logger.debug(
            "Gigachat request arguments (cert mode)",
            data={
                "arguments": arguments,
                "message_count": len(messages),
                "has_user": has_user_message,
            },
        )
        self._log_chat_progress(self.chat_turn(), model=model_name)

        try:
            # Headers for SSE streaming
            headers = {
                "Content-Type": "application/json",
                "Accept": EVENT_STREAM_HEADER,
            }

            # Get SSL configuration from _determine_auth_method
            # This ensures we use the same ssl_context that was created with proper CA bundle
            _, _, ssl_context_from_auth, cert_tuple_from_auth, _ = self._determine_auth_method()

            client_kwargs = {
                "base_url": self._base_url(),
                "headers": headers,
                "timeout": httpx.Timeout(60.0, read=60.0),
            }

            # Use ssl_context from _determine_auth_method if available (has proper CA bundle)
            # Otherwise fall back to _resolve_ssl_verify()
            if ssl_context_from_auth is not None:
                client_kwargs["verify"] = ssl_context_from_auth
            else:
                client_kwargs["verify"] = self._resolve_ssl_verify()

            # Add client certificates for mTLS authentication
            # Prefer cert_tuple from _determine_auth_method, fall back to _resolve_client_cert()
            if cert_tuple_from_auth is not None:
                client_kwargs["cert"] = cert_tuple_from_auth
            else:
                client_cert = self._resolve_client_cert()
                if client_cert:
                    client_kwargs["cert"] = client_cert

            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(
                    "/chat/completions",
                    json=arguments,
                )
                response.raise_for_status()

                final_response = await self._process_stream_cert(response, model_name)

                # Log response for debugging
                self.logger.debug(
                    "Gigachat cert mode: received response",
                    data={
                        "has_choices": hasattr(final_response, "choices")
                        and final_response.choices,
                        "has_content": hasattr(final_response, "choices")
                        and final_response.choices
                        and hasattr(final_response.choices[0], "message")
                        and hasattr(final_response.choices[0].message, "content"),
                        "content_length": len(
                            getattr(final_response.choices[0].message, "content", "")
                        )
                        if hasattr(final_response, "choices")
                        and final_response.choices
                        and hasattr(final_response.choices[0], "message")
                        else 0,
                    },
                )

        except httpx.HTTPStatusError as error:
            self.logger.error("HTTP error during Gigachat completion (cert mode)", exc_info=error)
            return self._stream_failure_response(error, model_name)
        except Exception as error:  # pragma: no cover - unexpected runtime failure
            self.logger.error("Error during Gigachat completion (cert mode)", exc_info=error)
            return self._stream_failure_response(error, model_name)

        # Usage tracking: we don't have TurnUsage.from_gigachat in this version,
        # but we can still forward raw usage for consistency if present.
        if (
            hasattr(final_response, "usage")
            and final_response.usage
            and not isinstance(final_response, BaseException)
        ):
            try:
                usage_data = final_response.usage or {}
                input_tokens = int(usage_data.get("prompt_tokens", 0))
                output_tokens = int(usage_data.get("completion_tokens", 0))
                total_tokens = int(usage_data.get("total_tokens", input_tokens + output_tokens))

                turn_usage = TurnUsage(
                    provider=Provider.GIGACHAT,
                    model=model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    raw_usage=usage_data,
                )
                self._finalize_turn_usage(turn_usage)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning("Failed to track Gigachat usage (cert mode): %s", exc)

        choice = final_response.choices[0]
        message_obj = choice.message

        # Extract content from response
        content = getattr(message_obj, "content", None)
        if content:
            response_blocks.append(TextContent(type="text", text=content))
        else:
            # If no content, log warning but still return a valid response
            # This ensures the turn is saved to history even if response is empty
            self.logger.warning(
                "GigaChat returned empty content in response",
                data={"model": model_name, "finish_reason": getattr(choice, "finish_reason", None)},
            )
            response_blocks.append(TextContent(type="text", text=""))

        stop_reason = LlmStopReason.END_TURN
        requested_tool_calls: Optional[Dict[str, CallToolRequest]] = None

        if getattr(message_obj, "tool_calls", None):
            requested_tool_calls = {}
            stop_reason = LlmStopReason.TOOL_USE
            for tool_call in message_obj.tool_calls:
                arguments = (
                    {}
                    if not tool_call.function.arguments
                    else from_json(tool_call.function.arguments, allow_partial=True)
                )
                requested_tool_calls[tool_call.id] = CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name=tool_call.function.name, arguments=arguments),
                )

        if request_params.use_history:
            # Update diagnostic history (provider's self.history is diagnostic only)
            # The actual conversation history is managed by the agent (_message_history)
            # Find the last user message from the full conversation
            last_user_message = None
            if message:
                # Find the last user message in the conversation
                for msg in reversed(message):
                    if msg.get("role") == "user":
                        last_user_message = msg
                        break

            if last_user_message:
                # Convert assistant response to provider format
                assistant_message_dict = {
                    "role": "assistant",
                    "content": message_obj.content if getattr(message_obj, "content", None) else "",
                }
                # Update diagnostic history (for debugging/inspection only)
                existing_history = self.history.get(include_completion_history=True)
                if existing_history:
                    # Append last turn to existing diagnostic history
                    existing_history.append(last_user_message)
                    existing_history.append(assistant_message_dict)
                    self.history.set(existing_history)
                else:
                    # First turn: create diagnostic history with last user message and response
                    self.history.set([last_user_message, assistant_message_dict])

        self._log_chat_finished(model=self.default_request_params.model)

        return Prompt.assistant(
            *response_blocks,
            stop_reason=stop_reason,
            tool_calls=requested_tool_calls,
        )

    async def _process_stream_cert(self, response: httpx.Response, model: str) -> SimpleNamespace:
        """Process SSE streaming response from GigaChat (certificate mode)."""
        estimated_tokens = 0
        accumulated_content = ""
        finish_reason = None
        usage_data = None
        role = "assistant"
        tool_calls: Dict[int, Dict[str, Any]] = {}

        tool_started: Dict[int, Dict[str, Any]] = {}
        notified_tool_indices: set[int] = set()
        streams_arguments = False

        async for line in response.aiter_lines():
            if not line or not line.startswith("data: "):
                continue

            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break

            try:
                payload = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = payload.get("choices") or []
            if choices:
                choice = choices[0]
                delta = choice.get("delta") or {}

                content = delta.get("content")
                if content:
                    accumulated_content += content
                    estimated_tokens = self._update_streaming_progress(
                        content,
                        model,
                        estimated_tokens,
                    )

                finish_reason = choice.get("finish_reason") or finish_reason

                delta_fn_call = delta.get("function_call")
                if isinstance(delta_fn_call, dict):
                    index = 0
                    info = tool_calls.setdefault(
                        index,
                        {
                            "id": delta_fn_call.get("id") or f"call_{uuid.uuid4().hex}",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if delta_fn_call.get("name"):
                        info["function"]["name"] = delta_fn_call["name"]

                    if delta_fn_call.get("arguments"):
                        args_value = delta_fn_call["arguments"]
                        if isinstance(args_value, str):
                            info["function"]["arguments"] += args_value
                        else:
                            info["function"]["arguments"] += json.dumps(args_value)

                    tool_started[index] = {
                        "tool_name": info["function"]["name"] or "gigachat_function",
                        "tool_use_id": info["id"],
                        "streams_arguments": streams_arguments,
                    }
                    if index not in notified_tool_indices and info["function"]["name"]:
                        self._notify_tool_stream_listeners(
                            "start",
                            {
                                "tool_name": info["function"]["name"],
                                "tool_use_id": info["id"],
                                "index": index,
                                "streams_arguments": streams_arguments,
                            },
                        )
                        notified_tool_indices.add(index)

                msg = choice.get("message") or {}
                if isinstance(msg, dict):
                    role = msg.get("role", role)
                    fn_call = msg.get("function_call")
                    if isinstance(fn_call, dict):
                        info = tool_calls.setdefault(
                            0,
                            {
                                "id": fn_call.get("id") or f"call_{uuid.uuid4().hex}",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if fn_call.get("name"):
                            info["function"]["name"] = fn_call["name"]
                        if fn_call.get("arguments"):
                            args_value = fn_call["arguments"]
                            if isinstance(args_value, str):
                                info["function"]["arguments"] = args_value
                            else:
                                info["function"]["arguments"] = json.dumps(args_value)

                if choice.get("finish_reason") == "tool_calls":
                    for index, info in tool_started.items():
                        self._notify_tool_stream_listeners(
                            "stop",
                            {
                                "tool_name": info.get("tool_name"),
                                "tool_use_id": info.get("tool_use_id"),
                                "index": index,
                                "streams_arguments": info.get("streams_arguments", False),
                            },
                        )

            if "usage" in payload:
                usage_data = payload["usage"]

        message_ns = SimpleNamespace()
        message_ns.content = accumulated_content
        message_ns.role = role

        if tool_calls:
            message_ns.tool_calls = [
                SimpleNamespace(
                    id=info["id"],
                    type=info["type"],
                    function=SimpleNamespace(
                        name=info["function"]["name"],
                        arguments=info["function"]["arguments"],
                    ),
                )
                for info in tool_calls.values()
                if info["function"]["name"]
            ]

        final_completion = SimpleNamespace()
        final_completion.choices = [SimpleNamespace()]
        final_completion.choices[0].message = message_ns
        final_completion.choices[0].finish_reason = finish_reason
        final_completion.usage = usage_data

        if usage_data:
            completion_tokens = usage_data.get("completion_tokens", estimated_tokens)
            token_str = str(completion_tokens).rjust(5)
            data = {
                "progress_action": ProgressAction.STREAMING,
                "model": model,
                "agent_name": self.name,
                "chat_turn": self.chat_turn(),
                "details": token_str.strip(),
            }
            self.logger.info("Gigachat streaming complete (cert mode)", data=data)

        return final_completion

    # ---------------------------------------------------------------------
    # Prompt application (branch between OpenAI-compatible and cert modes)
    # ---------------------------------------------------------------------

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List[PromptMessageExtended],
        request_params: Optional[RequestParams] = None,
        tools: Optional[List[Tool]] = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        """
        Route prompt handling based on auth method:
        - Certificate auth: use legacy direct HTTPX path (proven to work on local machine)
        - API key auth: use OpenAI-compatible base implementation
        """
        if self._use_cert_auth():
            # Certificate-based path - используем ту же логику, что и базовый класс
            # Вся история уже передана в multipart_messages, не нужно разделять на prior/last
            effective_params = self.get_request_params(request_params)
            last_message = multipart_messages[-1]

            # If the last message is from the assistant, no inference required
            if last_message.role == "assistant":
                return last_message

            # Convert ALL messages directly (same as base class)
            # System prompt will be added separately in _gigachat_completion_cert
            converted_messages: List[Dict[str, Any]] = []
            for message in multipart_messages:
                if message.role == "system":
                    # system подсказку добавляем отдельно в _gigachat_completion_cert
                    continue
                converted_messages.extend(self._convert_to_gigachat(message))

            # Fallback if conversion returned empty (same as base class)
            if not converted_messages:
                converted_messages = [{"role": "user", "content": ""}]

            # Pass ALL messages to _gigachat_completion_cert (no history separation)
            return await self._gigachat_completion_cert(converted_messages, effective_params, tools)

        # API key / OAuth path: use OpenAI-compatible implementation
        return await super()._apply_prompt_provider_specific(
            multipart_messages,
            request_params,
            tools,
            is_template,
        )

    # ---------------------------------------------------------------------
    # Request construction helpers (cert mode)
    # ---------------------------------------------------------------------

    def _prepare_api_request_cert(
        self,
        messages: List[Dict[str, Any]],
        functions: Optional[List[Dict[str, Any]]],
        request_params: RequestParams,
    ) -> Dict[str, Any]:
        base_args: Dict[str, Any] = {
            "model": self.default_request_params.model,
            "messages": messages,
            "stream": True,
        }

        if functions and not self._conversation_has_tool_results(messages):
            base_args["functions"] = functions

        if request_params.maxTokens:
            base_args["max_tokens"] = request_params.maxTokens

        if request_params.stopSequences:
            base_args["stop"] = request_params.stopSequences

        # Reuse base helper to merge provider-specific arguments, but with our exclude set
        arguments = self.prepare_provider_arguments(
            base_args,
            request_params,
            self.GIGACHAT_EXCLUDE_FIELDS.union(self.BASE_EXCLUDE_FIELDS),
        )

        return arguments

    def _prepare_functions_cert(
        self, tools: Optional[List[Tool]]
    ) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None

        functions: list[dict[str, Any]] = []
        for tool in tools:
            params = self._sanitize_function_schema(self.adjust_schema(tool.inputSchema))
            functions.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": params,
                }
            )
        return functions

    # ---------------------------------------------------------------------
    # Message conversions / schema helpers (cert mode)
    # ---------------------------------------------------------------------

    def _convert_to_gigachat(self, message: PromptMessageExtended) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []

        if message.tool_results:
            tool_texts = []
            for tool_call_id, tool_result in message.tool_results.items():
                result_text = self._extract_text_content(tool_result.content)
                if result_text:
                    tool_texts.append(f"Tool result ({tool_call_id}): {result_text}")
            if tool_texts:
                converted.append({"role": "user", "content": "\n".join(tool_texts)})

        content_text = self._extract_text_content(message.content)
        if not content_text:
            return converted

        if message.role in {"system", "assistant", "user"}:
            converted.append({"role": message.role, "content": content_text})
        elif message.role == "function":
            converted.append({"role": "function", "content": content_text})

        return converted

    def _extract_text_content(self, content: Iterable[ContentBlock]) -> str:
        parts: List[str] = []
        for block in content:
            if isinstance(block, TextContent):
                parts.append(block.text)
        return "".join(parts)

    def adjust_schema(self, input_schema: Dict[str, Any]) -> Dict[str, Any]:
        if "properties" in input_schema:
            return input_schema
        result = dict(input_schema)
        result["properties"] = {}
        return result

    def _sanitize_function_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(schema, dict):
            return {"type": "object", "properties": {}}

        clean: Dict[str, Any] = {"type": "object", "properties": {}}
        schema_type = schema.get("type", "object")
        clean["type"] = "object" if schema_type != "object" else schema_type

        properties = schema.get("properties") or {}
        allowed_keys = {
            "type",
            "description",
            "enum",
            "minimum",
            "maximum",
            "minLength",
            "maxLength",
            "items",
            "format",
            "nullable",
            "default",
        }
        for prop_name, prop_value in properties.items():
            if not isinstance(prop_value, dict):
                continue
            filtered = {k: v for k, v in prop_value.items() if k in allowed_keys}
            if "type" not in filtered:
                filtered["type"] = "string"
            clean["properties"][prop_name] = filtered

        required = schema.get("required")
        if isinstance(required, list):
            clean["required"] = [str(item) for item in required if isinstance(item, str)]

        if isinstance(schema.get("description"), str):
            clean["description"] = schema["description"]

        return clean

    def _conversation_has_tool_results(self, messages: List[Dict[str, Any]]) -> bool:
        return any(msg.get("role") == "function" for msg in messages)

    # ---------------------------------------------------------------------
    # Error handling (shared)
    # ---------------------------------------------------------------------

    def _stream_failure_response(
        self,
        error: Exception,
        model_name: str,
    ) -> PromptMessageExtended:  # pragma: no cover - exercised via integration tests
        provider_label = (
            self.provider.value if isinstance(self.provider, Provider) else str(self.provider)
        )
        detail = getattr(error, "message", None) or str(error)
        detail = detail.strip() if isinstance(detail, str) else ""

        parts: List[str] = [f"{provider_label} request failed"]
        if model_name:
            parts.append(f"for model '{model_name}'")

        if hasattr(error, "response") and getattr(error, "response", None) is not None:
            status = getattr(error.response, "status_code", None)
            if status:
                parts.append(f"(status={status})")

        message = " ".join(parts)
        if detail:
            message = f"{message}: {detail}"

        user_summary = " ".join(message.split()) if message else ""
        if user_summary and len(user_summary) > 280:
            user_summary = user_summary[:277].rstrip() + "..."

        if user_summary:
            assistant_text = f"I hit an internal error while calling the model: {user_summary}"
            if not assistant_text.endswith((".", "!", "?")):
                assistant_text += "."
            assistant_text += " See fast-agent-error for additional details."
        else:
            assistant_text = (
                "I hit an internal error while calling the model; see fast-agent-error for details."
            )

        assistant_block = text_content(assistant_text)
        error_block = text_content(message)

        return PromptMessageExtended(
            role="assistant",
            content=[assistant_block],
            channels={FAST_AGENT_ERROR_CHANNEL: [error_block]},
            stop_reason=LlmStopReason.ERROR,
        )

    # ---------------------------------------------------------------------
    # Usage tracking provider fix (shared)
    # ---------------------------------------------------------------------

    def _finalize_turn_usage(self, turn_usage: TurnUsage) -> None:
        """
        Override to ensure usage tracking uses GIGACHAT provider instead of OPENAI.
        """
        # Fix the provider if it was set to OPENAI (from TurnUsage.from_openai)
        if turn_usage.provider == Provider.OPENAI:
            # Create a new TurnUsage with GIGACHAT provider but same usage data
            turn_usage = TurnUsage(
                provider=Provider.GIGACHAT,
                model=turn_usage.model,
                input_tokens=turn_usage.input_tokens,
                output_tokens=turn_usage.output_tokens,
                total_tokens=turn_usage.total_tokens,
                cache_usage=turn_usage.cache_usage,
                tool_use_tokens=turn_usage.tool_use_tokens,
                reasoning_tokens=turn_usage.reasoning_tokens,
                tool_calls=turn_usage.tool_calls,
                raw_usage=turn_usage.raw_usage,
                timestamp=turn_usage.timestamp,
            )
        # Call parent to add to accumulator
        super()._finalize_turn_usage(turn_usage)
