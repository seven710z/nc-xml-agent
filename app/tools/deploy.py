"""
Tool layer — Tool Calling + MCP protocol wrapper.

Tool Calling vs MCP:
    1. Tool Calling at its core: the LLM emits a structured "call intent" —
       it does NOT execute anything; it returns {tool_name, arguments} JSON that
       the host code executes. This decouples "model suggestion" from "real side
       effects", which is the crux of safety.
    2. Why MCP: without it, every tool is hardcoded into the agent and swapping
       one means editing code and redeploying. With MCP, tools are standalone
       services exposed over a uniform protocol; the agent only configures a URL,
       so adding/removing tools never touches the main flow — this is what
       "config-driven iteration" means.
    3. Dangerous ops (deploy) require human-in-the-loop: a tool may *propose* a
       deploy, but nothing reaches the CNC device until the confirm node signs off.
"""

from __future__ import annotations

import httpx

from app.core.config import settings
from app.core.schemas import DeployRequest, DeployResult


# ---------------------------------------------------------------------------
# Controlled deploy tool (dangerous op, requires human confirmation)
# ---------------------------------------------------------------------------
class DeployTool:
    """Deploys a validated XML fragment to a CNC device.

    Safety (C6):
      - `confirmed` must be True before anything is actually dispatched;
      - otherwise only a "pending" status is returned, and the graph's confirm
        node acts as a second line of defense.
    """

    name = "deploy_xml"
    description = "将校验通过的数控界面 XML 片段部署到目标数控设备。这是危险操作，需要人工确认。"
    parameters_schema = {
        "type": "object",
        "properties": {
            "xml": {"type": "string", "description": "要部署的 XML 片段"},
            "target_device": {"type": "string", "description": "目标数控设备标识"},
            "confirmed": {"type": "boolean", "description": "是否已通过人工确认"},
        },
        "required": ["xml", "target_device", "confirmed"],
    }

    async def run(self, req: DeployRequest) -> DeployResult:
        if not req.confirmed:
            return DeployResult(
                ok=False,
                message="未通过人工确认，已拦截下发（human-in-the-loop）",
                device=req.target_device,
            )

        # ===== Placeholder: low-level CNC deployment is out of scope =====
        # In production this dispatches to the device via the vendor SDK / MCP
        # service. I own only the "controlled call + confirmation gate" layer;
        # the hardware protocol is implemented by the hardware team.
        # =================================================================
        return DeployResult(ok=True, message="部署指令已下发（占位）", device=req.target_device)


# ---------------------------------------------------------------------------
# MCP tool-service client
# ---------------------------------------------------------------------------
class MCPClient:
    """A minimal MCP client for discovering and calling external tool services.

    MCP exposes tools over JSON-RPC. Two core capabilities implemented here:
      1. tools/list — dynamically discover server-side tools (basis of config iteration);
      2. tools/call — invoke a tool.
    Production would use the official MCP SDK; httpx is hand-rolled here to show
    that the protocol is essentially JSON-RPC.
    """

    def __init__(self, server_url: str = ""):
        self.server_url = server_url or settings.mcp_server_url

    async def list_tools(self) -> list[dict]:
        """Discover remote tools, with timeout + fallback (C4)."""
        if settings.app_mode == "mock":
            return [{"name": DeployTool.name, "description": DeployTool.description}]
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.server_url}/rpc",
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                )
            resp.raise_for_status()
            return resp.json().get("result", {}).get("tools", [])
        except Exception:
            # Fallback: MCP unavailability only affects remote tools, not the main flow.
            return []

    async def call(self, name: str, arguments: dict) -> dict:
        """Invoke a remote tool."""
        if settings.app_mode == "mock":
            return {"ok": True, "mock": True, "tool": name, "args": arguments}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{self.server_url}/rpc",
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                },
            )
        resp.raise_for_status()
        return resp.json().get("result", {})


deploy_tool = DeployTool()
mcp_client = MCPClient()
