import websockets
import json
import logging
import asyncio
from typing import Optional, List, Dict, Any
from src.logger import logger

class VTubeStudioModule:
    def __init__(self, host: str = "localhost", port: int = 8001):
        self.host = host
        self.port = port
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self.auth_token = None
        self.request_id_counter = 0

    def _get_request_id(self) -> str:
        self.request_id_counter += 1
        return f"Mikudayo-Script-{self.request_id_counter}"

    async def connect(self):
        uri = f"ws://{self.host}:{self.port}"
        try:
            self.websocket = await websockets.connect(uri)
            self.connected = True
            logger.info("Connected to VTube Studio WebSocket.")
            await self._authenticate()
        except Exception as e:
            logger.error(f"VTube Studio connection error: {e}. Is VTS running and API enabled?")
            self.connected = False

    async def _authenticate(self):
        if not self.websocket or not self.connected:
            return

        auth_token_request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": self._get_request_id(),
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": "AI Avatar (Python)",
                "pluginDeveloper": "Local AI User",
            }
        }
        await self.websocket.send(json.dumps(auth_token_request))
        response_str = await self.websocket.recv()
        response = json.loads(response_str)

        if response.get("messageType") == "AuthenticationTokenResponse" and response.get("data", {}).get("authenticationToken"):
            self.auth_token = response["data"]["authenticationToken"]
            logger.info(f"VTube Studio authentication token received: {self.auth_token[:10]}...")

            auth_request = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": self._get_request_id(),
                "messageType": "AuthenticationRequest",
                "data": {
                    "pluginName": "AI Avatar (Python)",
                    "pluginDeveloper": "Local AI User",
                    "authenticationToken": self.auth_token
                }
            }
            await self.websocket.send(json.dumps(auth_request))
            auth_response_str = await self.websocket.recv()
            auth_response = json.loads(auth_response_str)

            if auth_response.get("messageType") == "AuthenticationResponse" and auth_response.get("data", {}).get("authenticated"):
                logger.info("Successfully authenticated with VTube Studio.")
            else:
                logger.error(f"VTube Studio authentication failed: {auth_response.get('data', {}).get('reason')}")
                self.connected = False
                self.auth_token = None
        else:
            logger.error(f"Failed to get VTube Studio authentication token: {response.get('data', {}).get('reason')}")
            self.connected = False

    async def trigger_hotkey(self, hotkey_id: str):
        if not self.connected or not self.websocket or not self.auth_token or not hotkey_id:
            logger.warning(f"VTube Studio not ready to trigger hotkey '{hotkey_id}'.")
            return

        request = {
            "apiName": "VTubeStudioPublicAPI", "apiVersion": "1.0",
            "requestID": self._get_request_id(),
            "messageType": "HotkeyTriggerRequest",
            "data": {"hotkeyID": hotkey_id}
        }
        try:
            await self.websocket.send(json.dumps(request))
            logger.info(f"Sent HotkeyTriggerRequest for '{hotkey_id}'.")
        except Exception as e:
            logger.error(f"Error triggering VTube Studio hotkey '{hotkey_id}': {e}")
            if isinstance(e, websockets.exceptions.ConnectionClosed):
                self.connected = False

    async def inject_parameters(self, params: List[Dict[str, Any]]):
        if not self.connected or not self.websocket or not self.auth_token:
            return

        message = {
            "apiName": "VTubeStudioPublicAPI", "apiVersion": "1.0",
            "requestID": self._get_request_id(),
            "messageType": "InjectParameterDataRequest",
            "data": { "mode": "set", "parameterValues": params }
        }
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error injecting VTS parameters: {e}")
            if isinstance(e, websockets.exceptions.ConnectionClosed):
                self.connected = False

    async def close(self):
        if self.websocket and self.connected:
            logger.info("Closing VTube Studio WebSocket connection.")
            await self.websocket.close()
            self.connected = False
            self.websocket = None
