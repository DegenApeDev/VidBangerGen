from __future__ import annotations

import asyncio
import json
import math
import mimetypes
from pathlib import Path, PurePosixPath
from typing import Any, Awaitable, Callable
from urllib.parse import urlencode, urlparse
from uuid import uuid4

import httpx
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException


ProgressCallback = Callable[[float, str | None], Awaitable[None]]


class ComfyError(RuntimeError):
    pass


class ComfyTransportError(ComfyError):
    """A retryable ComfyUI HTTP/progress-channel failure."""


class ComfyOrphanedPromptError(ComfyError):
    """A queued prompt vanished from both ComfyUI's queue and history."""


def parse_history_entry(prompt_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for node_output in entry.get("outputs", {}).values():
        if not isinstance(node_output, dict):
            continue
        for media_list in node_output.values():
            if not isinstance(media_list, list):
                continue
            for item in media_list:
                if isinstance(item, dict) and item.get("type") == "output" and item.get("filename"):
                    files.append(
                        {
                            "filename": item["filename"],
                            "subfolder": item.get("subfolder", ""),
                            "type": item.get("type", "output"),
                        }
                    )

    prompt_field = entry.get("prompt")
    graph: dict[str, Any] = {}
    prompt_number = None
    metadata: dict[str, Any] = {}
    if isinstance(prompt_field, list):
        if len(prompt_field) > 0:
            prompt_number = prompt_field[0]
        if len(prompt_field) > 2 and isinstance(prompt_field[2], dict):
            graph = prompt_field[2]
        if len(prompt_field) > 3 and isinstance(prompt_field[3], dict):
            metadata = prompt_field[3]
    elif isinstance(prompt_field, dict):
        graph = prompt_field

    prompt_candidates: list[str] = []
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if node.get("class_type") == "CLIPTextEncode" and inputs.get("text"):
            prompt_candidates.append(str(inputs["text"]))
    # API-format graphs do not label positive vs negative conditioning. The authored
    # positive prompt is consistently the more descriptive CLIP text in LTX graphs.
    prompt_text = max(prompt_candidates, key=len, default="")

    status = entry.get("status", {})
    error = status.get("error_message")
    messages = status.get("messages", [])
    start_ms = finish_ms = None
    for message in messages:
        if not isinstance(message, list) or len(message) != 2 or not isinstance(message[1], dict):
            continue
        if message[0] == "execution_start":
            start_ms = message[1].get("timestamp")
        elif message[0] in ("execution_success", "execution_error", "execution_interrupted"):
            finish_ms = message[1].get("timestamp")
            if message[0] in ("execution_error", "execution_interrupted"):
                error = (
                    message[1].get("exception_message")
                    or message[1].get("exception_type")
                    or message[0]
                )
    create_ms = metadata.get("create_time")
    return {
        "prompt_id": prompt_id,
        "prompt_number": prompt_number,
        "status": "error" if error else ("done" if status.get("completed") else "running"),
        "error": error,
        "files": files,
        "prompt": prompt_text,
        "created_at_ms": create_ms,
        "started_at_ms": start_ms,
        "finished_at_ms": finish_ms,
        "elapsed_seconds": (
            round((finish_ms - start_ms) / 1000, 3)
            if isinstance(start_ms, (int, float)) and isinstance(finish_ms, (int, float))
            else None
        ),
    }


class ComfyClient:
    def __init__(
        self, base_url: str, timeout_seconds: int = 900,
        orphan_grace_seconds: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        # Comfy can remove a prompt from /queue before its /history entry is
        # committed, especially while SaveVideo is flushing a larger file.
        # Polling happens every three seconds, so keep a time-shaped bound while
        # retaining deterministic tests that replace asyncio.sleep.
        self.orphan_grace_checks = max(1, math.ceil(orphan_grace_seconds / 3.0))
        self.worker_id = urlparse(self.base_url).netloc.replace(":", "_")

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/system_stats", timeout=10)
            response.raise_for_status()
            return response.json()

    async def queue_state(self) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/queue", timeout=10)
            response.raise_for_status()
            value = response.json()
        return {
            "running": value.get("queue_running") or [],
            "pending": value.get("queue_pending") or [],
        }

    async def prompt_in_queue(self, prompt_id: str) -> bool:
        state = await self.queue_state()
        return prompt_id in (
            self._queued_prompt_ids(state["running"])
            | self._queued_prompt_ids(state["pending"])
        )

    async def release_managed_models(self) -> None:
        """Release only an explicitly managed worker's transient model cache."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/free",
                json={"unload_models": True, "free_memory": True},
                timeout=30,
            )
        if response.status_code not in (200, 201):
            raise ComfyError(
                f"Managed ComfyUI worker refused model release ({response.status_code})"
            )

    @staticmethod
    def _queued_prompt_ids(entries: list[Any]) -> set[str]:
        values: set[str] = set()
        for entry in entries:
            if isinstance(entry, (list, tuple)) and len(entry) > 1:
                values.add(str(entry[1]))
            elif isinstance(entry, dict) and entry.get("prompt_id"):
                values.add(str(entry["prompt_id"]))
        return values

    async def cancel_prompt(self, prompt_id: str) -> str:
        """Cancel only the named prompt on an application-owned worker."""
        state = await self.queue_state()
        if prompt_id in self._queued_prompt_ids(state["pending"]):
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/queue", json={"delete": [prompt_id]}, timeout=10
                )
            if response.status_code not in (200, 201):
                raise ComfyError(
                    f"ComfyUI refused queued prompt deletion ({response.status_code})"
                )
            return "deleted"
        if prompt_id in self._queued_prompt_ids(state["running"]):
            await self.interrupt()
            return "interrupted"
        return "not-found"

    async def queue(self, graph: dict[str, Any], client_id: str | None = None) -> str:
        payload: dict[str, Any] = {"prompt": graph}
        if client_id:
            payload["client_id"] = client_id
        response: httpx.Response | None = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/prompt", json=payload, timeout=30
                    )
                break
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                if attempt == 2:
                    raise ComfyTransportError(
                        f"Could not reach the prompt endpoint at {self.base_url} "
                        "after 3 attempts. This is a worker connection problem, "
                        "not a GPU-busy condition."
                    ) from exc
                await asyncio.sleep(1.5 * (attempt + 1))
        if response is None:  # pragma: no cover - defensive; loop always sets or raises
            raise ComfyTransportError(f"Prompt request to {self.base_url} returned no response")
        if response.status_code != 200:
            raise ComfyError(f"ComfyUI rejected workflow ({response.status_code}): {response.text[:1000]}")
        data = response.json()
        if not data.get("prompt_id"):
            raise ComfyError("ComfyUI response did not contain prompt_id")
        return str(data["prompt_id"])

    async def history(self, prompt_id: str) -> dict[str, Any] | None:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/history/{prompt_id}", timeout=15)
        if response.status_code != 200:
            raise ComfyTransportError(f"History request failed with {response.status_code}")
        entry = response.json().get(prompt_id)
        return parse_history_entry(prompt_id, entry) if isinstance(entry, dict) else None

    async def all_history(self) -> list[dict[str, Any]]:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/history", timeout=30)
        response.raise_for_status()
        return [
            parse_history_entry(prompt_id, entry)
            for prompt_id, entry in response.json().items()
            if isinstance(entry, dict)
        ]

    async def upload(self, path: Path, remote_name: str | None = None) -> str:
        safe_name = remote_name or f"vbg_{uuid4().hex}_{path.name}"
        safe_name = PurePosixPath(safe_name).name
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        response: httpx.Response | None = None
        # A worker can briefly refuse new TCP connections while ComfyUI is
        # loading/offloading large LTX components. That is independent of GPU
        # capacity, and a second attempt is normally enough. Reopen the file for
        # each request so multipart retries always start at byte zero.
        for attempt in range(3):
            try:
                async with httpx.AsyncClient() as client:
                    with path.open("rb") as handle:
                        response = await client.post(
                            f"{self.base_url}/upload/image",
                            files={"image": (safe_name, handle, mime)},
                            data={"overwrite": "false", "type": "input"},
                            timeout=60,
                        )
                break
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                if attempt == 2:
                    raise ComfyTransportError(
                        f"Could not reach the upload endpoint at {self.base_url} "
                        "after 3 attempts. This is a worker connection problem, "
                        "not a GPU-busy condition."
                    ) from exc
                await asyncio.sleep(1.5 * (attempt + 1))

        if response is None:  # pragma: no cover - defensive; loop always sets or raises
            raise ComfyTransportError(f"Upload to {self.base_url} did not return a response")
        if response.status_code not in (200, 201):
            hint = (
                " Check that the ComfyUI runtime user can write its configured input directory."
                if response.status_code >= 500 else ""
            )
            raise ComfyError(
                f"Upload failed ({response.status_code}): {response.text[:500]}{hint}"
            )
        data = response.json()
        subfolder = data.get("subfolder", "")
        return f"{subfolder}/{data.get('name', safe_name)}".lstrip("/")

    async def download(self, artifact: dict[str, Any], destination: Path) -> None:
        params = urlencode(
            {
                "filename": artifact["filename"],
                "subfolder": artifact.get("subfolder", ""),
                "type": artifact.get("type", "output"),
            }
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", f"{self.base_url}/view?{params}", timeout=120) as response:
                if response.status_code != 200:
                    raise ComfyError(f"Download failed with {response.status_code}")
                with destination.open("wb") as handle:
                    async for chunk in response.aiter_bytes():
                        handle.write(chunk)

    async def interrupt(self) -> None:
        async with httpx.AsyncClient() as client:
            await client.post(f"{self.base_url}/interrupt", timeout=10)

    def websocket_url(self, client_id: str) -> str:
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return f"{scheme}://{parsed.netloc}{parsed.path}/ws?{urlencode({'clientId': client_id})}"

    async def queue_and_wait(
        self,
        graph: dict[str, Any],
        client_id: str,
        on_queued: Callable[[str], Awaitable[None]] | None = None,
        progress: ProgressCallback | None = None,
        cancelled: Callable[[], bool] | None = None,
        allow_interrupt: bool = False,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Queue with a subscribed socket so sampler progress is reported in real time."""
        prompt_id: str | None = None
        try:
            async with connect(
                self.websocket_url(client_id), open_timeout=10, close_timeout=3, max_size=None
            ) as socket:
                prompt_id = await self.queue(graph, client_id=client_id)
                if on_queued:
                    await on_queued(prompt_id)
                loop = asyncio.get_running_loop()
                timeout = timeout_seconds or self.timeout_seconds
                deadline = loop.time() + timeout
                highest = 0.01
                missing_checks = 0
                while loop.time() < deadline:
                    if cancelled and cancelled() and allow_interrupt:
                        await self.cancel_prompt(prompt_id)
                        raise asyncio.CancelledError
                    try:
                        message = await asyncio.wait_for(socket.recv(), timeout=3)
                    except TimeoutError:
                        entry = await self.history(prompt_id)
                        if entry and entry["status"] == "done":
                            if progress:
                                await progress(1.0, None)
                            return entry
                        if entry and entry["status"] == "error":
                            raise ComfyError(entry.get("error") or "ComfyUI execution failed")
                        if entry is None:
                            state = await self.queue_state()
                            present = prompt_id in (
                                self._queued_prompt_ids(state["running"])
                                | self._queued_prompt_ids(state["pending"])
                            )
                            missing_checks = 0 if present else missing_checks + 1
                            if missing_checks >= self.orphan_grace_checks:
                                raise ComfyOrphanedPromptError(
                                    f"Remote prompt {prompt_id} disappeared from queue and history"
                                )
                        else:
                            missing_checks = 0
                        continue
                    if not isinstance(message, str):
                        continue
                    event = json.loads(message)
                    data = event.get("data", {})
                    event_prompt = data.get("prompt_id")
                    if event_prompt and event_prompt != prompt_id:
                        continue
                    event_type = event.get("type")
                    if event_type == "progress":
                        maximum = max(1, int(data.get("max") or 1))
                        raw = max(0.0, min(1.0, float(data.get("value") or 0) / maximum))
                        highest = max(highest, min(0.95, 0.05 + raw * 0.9))
                        if progress:
                            await progress(highest, str(data.get("node") or "") or None)
                    elif event_type == "executing":
                        node = data.get("node")
                        if progress and node is not None:
                            await progress(highest, str(node))
                        if node is None and event_prompt == prompt_id:
                            for _ in range(10):
                                entry = await self.history(prompt_id)
                                if entry and entry["status"] == "done":
                                    if progress:
                                        await progress(1.0, None)
                                    return entry
                                await asyncio.sleep(0.25)
                    elif event_type in ("execution_error", "execution_interrupted"):
                        raise ComfyError(str(data.get("exception_message") or event_type))
                raise ComfyError(f"Generation timed out after {timeout} seconds")
        except asyncio.CancelledError:
            raise
        except (
            OSError, WebSocketException, json.JSONDecodeError, httpx.HTTPError,
            ComfyTransportError,
        ) as socket_error:
            # Preserve execution if the progress channel drops. Queue only if it was never queued.
            if prompt_id is None:
                prompt_id = await self.queue(graph, client_id=client_id)
                if on_queued:
                    await on_queued(prompt_id)
            if progress:
                await progress(0.02, f"websocket fallback: {type(socket_error).__name__}")
            return await self.wait_for_completion(
                prompt_id, progress=progress, cancelled=cancelled,
                allow_interrupt=allow_interrupt, timeout_seconds=timeout_seconds,
            )

    async def wait_for_completion(
        self,
        prompt_id: str,
        progress: ProgressCallback | None = None,
        cancelled: Callable[[], bool] | None = None,
        allow_interrupt: bool = False,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        timeout = timeout_seconds or self.timeout_seconds
        deadline = loop.time() + timeout
        tick = 0
        missing_checks = 0
        while loop.time() < deadline:
            if cancelled and cancelled() and allow_interrupt:
                await self.cancel_prompt(prompt_id)
                raise asyncio.CancelledError
            try:
                entry = await self.history(prompt_id)
            except (httpx.HTTPError, ComfyTransportError):
                # ComfyUI's HTTP loop can briefly stall while a large VAE decode or
                # encoder flush is finishing.  The remote prompt remains valid, so
                # keep polling it instead of failing (or accidentally re-queuing) a
                # completed generation.
                tick += 1
                if progress:
                    await progress(min(0.95, 0.02 + tick * 0.01), "history retry")
                await asyncio.sleep(3)
                continue
            if entry:
                missing_checks = 0
                if entry["status"] == "done":
                    if progress:
                        await progress(1.0, None)
                    return entry
                if entry["status"] == "error":
                    raise ComfyError(entry.get("error") or "ComfyUI execution failed")
            else:
                try:
                    state = await self.queue_state()
                except (httpx.HTTPError, ComfyTransportError):
                    # If the queue itself cannot be inspected, absence is not
                    # evidence that ComfyUI lost the prompt.
                    missing_checks = 0
                else:
                    present = prompt_id in (
                        self._queued_prompt_ids(state["running"])
                        | self._queued_prompt_ids(state["pending"])
                    )
                    missing_checks = 0 if present else missing_checks + 1
                    if missing_checks >= self.orphan_grace_checks:
                        raise ComfyOrphanedPromptError(
                            f"Remote prompt {prompt_id} disappeared from queue and history"
                        )
            tick += 1
            if progress:
                # Fallback when the live progress socket is unavailable.
                await progress(min(0.95, 0.02 + tick * 0.01), None)
            await asyncio.sleep(3)
        raise ComfyError(f"Generation timed out after {timeout} seconds")
