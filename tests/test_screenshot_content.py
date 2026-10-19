"""Screenshot wire-content tests without browser or cleanup side effects."""

import ast
import asyncio
import base64
import hashlib
import io
import json
import os
import sys
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from typing import Optional, Union
from unittest.mock import patch

from fastmcp import Client, FastMCP
from fastmcp.client.transports import PythonStdioTransport
from fastmcp.tools.tool import ToolResult
from mcp.types import CallToolResult, ImageContent, TextContent
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class ScreenshotContentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.captures = []
        buffer = io.BytesIO()
        Image.new("RGBA", (96, 64), (30, 60, 90, 128)).save(buffer, format="PNG")
        self.png = buffer.getvalue()
        self.capture_error = None
        self.capture_started = None

    def server(self):
        # Load the exact production function without importing server.py's global
        # cleanup/browser objects. Registration and client serialization are real.
        tree = ast.parse((ROOT / "src/server.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                        and n.name == "take_screenshot")
        mcp = FastMCP("screenshot-wire-test")
        owner = self

        class Tab:
            async def save_screenshot(self, path, **options):
                owner.captures.append((Path(path), options))
                Path(path).write_bytes(owner.png)
                if owner.capture_error:
                    raise owner.capture_error
                if owner.capture_started:
                    owner.capture_started.set()
                    await asyncio.Event().wait()

        class Manager:
            async def get_tab(self, instance_id):
                return Tab() if instance_id == "owned" else None

        namespace = dict(
            __name__="screenshot_test", section_tool=lambda _: mcp.tool,
            browser_manager=Manager(), Path=Path, Optional=Optional, Union=Union,
            ToolResult=ToolResult, ImageContent=ImageContent, TextContent=TextContent,
            tempfile=tempfile, os=os, base64=base64, json=json,
        )
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / "src/server.py"), "exec"), namespace)
        return mcp

    def call(self, **kwargs):
        async def run():
            async with Client(self.server()) as client:
                result = await client.call_tool("take_screenshot", {"instance_id": "owned", **kwargs}, raise_on_error=False)
                # Exercise the actual MCP CallToolResult JSON representation.
                wire = CallToolResult(content=result.content, structuredContent=result.structured_content,
                                      isError=result.is_error).model_dump_json()
                return CallToolResult.model_validate_json(wire)
        return asyncio.run(run())

    def test_png_is_an_image_on_mcp_wire_with_complete_hash_metadata(self):
        result = self.call()
        self.assertFalse(result.isError)
        self.assertEqual(len(result.content), 2)
        image = result.content[0]
        self.assertIsInstance(image, ImageContent)
        self.assertEqual(image.mimeType, "image/png")
        data = base64.b64decode(image.data, validate=True)
        self.assertEqual(Image.open(io.BytesIO(data)).size, (96, 64))
        self.assertEqual(result.structuredContent["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(result.structuredContent["bytes"], len(data))
        self.assertTrue(result.structuredContent["complete"])
        self.assertFalse(self.captures[0][0].exists())
        self.assertNotIn("file_path", result.structuredContent)
        self.assertEqual(json.loads(result.content[1].text), result.structuredContent)

    def test_jpg_alias_has_correct_mime_and_flattens_transparency(self):
        result = self.call(format="jpg", full_page=True)
        self.assertFalse(result.isError)
        self.assertEqual(result.content[0].mimeType, "image/jpeg")
        image = Image.open(io.BytesIO(base64.b64decode(result.content[0].data)))
        self.assertEqual(image.format, "JPEG")
        self.assertEqual(image.mode, "RGB")
        self.assertTrue(all(abs(a - b) < 4 for a, b in zip(image.getpixel((30, 30)), (142, 157, 172))))
        self.assertTrue(self.captures[0][1]["full_page"])

    def test_stdio_client_receives_image_without_server_file_access(self):
        async def run():
            transport = PythonStdioTransport(
                Path(__file__), args=["--serve-fixture"],
                python_cmd=sys.executable, cwd=self.directory.name, keep_alive=False,
            )
            async with Client(transport) as client:
                self.assertIn("take_screenshot", [tool.name for tool in await client.list_tools()])
                return await client.call_tool("take_screenshot", {"instance_id": "owned"})

        result = asyncio.run(run())
        self.assertEqual(len(result.content), 2)
        self.assertIsInstance(result.content[0], ImageContent)
        image = base64.b64decode(result.content[0].data, validate=True)
        self.assertEqual(Image.open(io.BytesIO(image)).size, (96, 64))
        self.assertEqual(result.structured_content["sha256"], hashlib.sha256(image).hexdigest())

    def test_near_limit_image_crosses_stdio_without_truncation(self):
        async def run():
            transport = PythonStdioTransport(
                Path(__file__), args=["--serve-large-fixture"],
                python_cmd=sys.executable, cwd=self.directory.name, keep_alive=False,
            )
            async with Client(transport) as client:
                return await client.call_tool("take_screenshot", {"instance_id": "owned"})
        result = asyncio.run(run())
        image = base64.b64decode(result.content[0].data, validate=True)
        self.assertGreater(len(image), 8 * 1024 * 1024 * .98)
        self.assertLessEqual(len(image), 8 * 1024 * 1024)
        self.assertEqual(Image.open(io.BytesIO(image)).size, (1660, 1660))
        self.assertEqual(result.structured_content["sha256"], hashlib.sha256(image).hexdigest())

    def test_explicit_file_export_stays_available(self):
        target = Path(self.directory.name) / "exports" / "capture.png"
        result = self.call(file_path=str(target))
        self.assertFalse(result.isError)
        self.assertTrue(target.is_file())
        self.assertIn(str(target), result.content[0].text)
        self.assertNotIn("Read tool", result.content[0].text)

    def test_invalid_format_fails_before_capture(self):
        self.assertTrue(self.call(format="gif").isError)
        self.assertEqual(self.captures, [])

    def test_oversized_capture_fails_and_removes_temporary_file(self):
        self.png = b"x" * (16 * 1024 * 1024 + 1)
        result = self.call()
        self.assertTrue(result.isError)
        self.assertIn("inline capture limit", result.content[0].text)
        self.assertFalse(self.captures[0][0].exists())

    def test_excessive_pixels_are_rejected_before_decode(self):
        header = self.png[12:16] + struct.pack(">II", 4097, 4097) + self.png[24:29]
        self.png = self.png[:12] + header + struct.pack(">I", zlib.crc32(header)) + self.png[33:]
        result = self.call()
        self.assertTrue(result.isError)
        self.assertIn("dimensions exceed inline limit", result.content[0].text)
        self.assertFalse(self.captures[0][0].exists())

    def test_capture_failure_removes_temporary_file(self):
        self.capture_error = RuntimeError("Synthetic capture failure")
        self.assertTrue(self.call().isError)
        self.assertFalse(self.captures[0][0].exists())

    def test_cancelled_capture_removes_temporary_file(self):
        async def run():
            self.capture_started = asyncio.Event()
            async with Client(self.server()) as client:
                task = asyncio.create_task(client.call_tool("take_screenshot", {"instance_id": "owned"}))
                await asyncio.wait_for(self.capture_started.wait(), 2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        asyncio.run(run())
        self.assertFalse(self.captures[0][0].exists())

    def test_encoded_output_limit_is_not_silently_spilled_or_truncated(self):
        # A synthetic encoder result tests the precise post-compression boundary.
        # Input remains a valid small capture; only output encoding is mocked.
        self.png = ScreenshotContentTests.small_png()
        with patch.object(Image.Image, "save", lambda _, stream, **kw: stream.write(b"x" * (8 * 1024 * 1024 + 1))):
            self.assertTrue(self.call().isError)
        self.assertFalse(self.captures[0][0].exists())

    @staticmethod
    def small_png():
        buffer = io.BytesIO()
        Image.new("RGB", (1, 1)).save(buffer, format="PNG")
        return buffer.getvalue()


if __name__ == "__main__":
    if sys.argv[1:] in (["--serve-fixture"], ["--serve-large-fixture"]):
        fixture = ScreenshotContentTests()
        fixture.setUp()
        if sys.argv[1] == "--serve-large-fixture":
            buffer = io.BytesIO()
            Image.frombytes("RGB", (1660, 1660), os.urandom(1660 * 1660 * 3)).save(buffer, format="PNG")
            fixture.png = buffer.getvalue()
        try:
            fixture.server().run(show_banner=False)
        finally:
            fixture.doCleanups()
    else:
        unittest.main()
