import sys
import inspect
import json
import asyncio
import aiohttp
from aiohttp import WSCloseCode

class WSConnection(object):
    def __init__(self, ws, loop, on_message_callbacks=None):
        self.ws = ws
        self.loop = loop
        self.alive = False
        self.send_queue = asyncio.Queue(loop=self.loop)
        self.on_message_callbacks = on_message_callbacks if on_message_callbacks is not None else list()

    async def close(self):
        if self.ws is not None and not self.ws.closed:
            await self.ws.close(code=WSCloseCode.GOING_AWAY, message='CLOSE CONNECTION')
        self.alive = False

    async def send(self, data):
        try:
            if self.ws is None or self.alive == False or self.ws.closed:
                return
            self.ws.send_str(json.dumps(data))
        except Exception as e:
            print(f"ERROR SENDING: {e.__class__.__name__}: {e}", file=sys.stderr)
            pass

    async def receiving(self):
        try:
            self.alive = True
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    msg = json.loads(msg.data)
                    for callback in self.on_message_callbacks:
                        ans = callback(msg)
                        if inspect.isawaitable(ans):
                            await ans
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print('ws connection closed with exception %s' %
                          self.ws.exception(), file=sys.stderr)
                    self.alive = False
                    raise self.ws.exception()
        finally:
            self.alive = False

    def run(self):
        return asyncio.ensure_future(self.receiving(), loop=self.loop)

    def add_callback(self, callback):
        self.on_message_callbacks.append(callback)

    def remove_callback(self, callback):
        self.on_message_callbacks.remove(callback)


class WSChannelReceiver(object):
    def __init__(self, conn, i_version):
        self.i_version = i_version
        self.conn = conn
        self.loop = self.conn.loop
        self.receive_buffer = asyncio.Queue()
        self.alive = True
        self.conn.add_callback(self.recv_on_message)
        self.bg_fetch_job = asyncio.ensure_future(self.bg_fetch(), loop=self.loop)

    def close(self):
        self.alive = False

    async def get(self):
        return await self.receive_buffer.get()

    async def bg_fetch(self):
        while self.alive == True:
            await asyncio.sleep(2)
            await self.fetch()
        # wait unsended message to be send
        await asyncio.sleep(5)
        await self.conn.close()

    async def fetch(self):
        await self.conn.send({
            "type": "FETCH",
            "version": self.i_version
        })

    async def recv_on_message(self, msg):
        if msg["type"] == "TOUCH":
            await self.fetch()
        elif msg["type"] == "MSG":
            if self.i_version < msg["version"]:
                self.i_version = msg["version"]
                await self.receive_buffer.put(msg["body"])
                asyncio.ensure_future(self.fetch(), loop=self.loop)


class WSChannelSender(object):
    def __init__(self, conn, o_version, buffer_size=64):
        self.o_version = o_version
        self.conn = conn
        self.buffer_size = buffer_size
        self.send_buffer = asyncio.Queue(maxsize=buffer_size)
        self.current = None
        self.loop = self.conn.loop
        self.alive = True
        self.conn.add_callback(self.send_on_message)

    async def put(self, data):
        await self.send_buffer.put(data)
        await self.conn.send({
            "type": "TOUCH"
        })

    def put_nowait(self, data):
        self.send_buffer.put_nowait(data)
        asyncio.ensure_future(self.conn.send({"type": "TOUCH"}), loop=self.loop)

    async def send(self):
        await self.conn.send({
            "type": "MSG",
            "version": self.o_version,
            "body": self.current
        })

    async def send_on_message(self, msg):
        if msg["type"] == "FETCH":
            if msg["version"] > self.o_version == 0:
                print("SOME MESSAGE MAY BE LOST, You are sending from a new started server", file=sys.stderr)
                self.o_version = msg["version"]
            if msg["version"] < self.o_version:
                await self.send()
            elif msg["version"] == self.o_version:
                if not self.send_buffer.empty():
                    self.current = self.send_buffer.get_nowait()
                    self.o_version += 1
                    await self.send()
            else:
                print(f"o_version alwasy >= version. GOT: {self.o_version} > {msg['version']}: {msg}", file=sys.stderr)


class WSChannel(WSChannelReceiver, WSChannelSender):
    def __init__(self, conn, i_version, o_version, buffer_size=64):
        WSChannelReceiver.__init__(self, conn, i_version)
        WSChannelSender.__init__(self, conn, o_version, buffer_size)

    def close(self):
        WSChannelReceiver.close(self)
