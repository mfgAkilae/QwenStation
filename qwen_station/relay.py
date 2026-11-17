"""Windows: lets other devices reach the API of a model that runs inside WSL.

WSL forwards 127.0.0.1:<port> of Windows into the distribution, and nothing else. This relay listens on the
address the user chose and passes every connection on to 127.0.0.1:<port>, bytes unchanged. It needs no
administrator rights and does not depend on the address of the WSL network, which changes with every restart
of Windows (a `netsh portproxy` rule would have to be rewritten each time).
"""

from __future__ import annotations

import ipaddress
import socket
import threading

from .config import LOCAL_HOST

CHUNK = 65536
CONNECT_TIMEOUT = 5.0


class Relay:
    def __init__(self, host: str, port: int, target_port: int | None = None, target_host: str = LOCAL_HOST) -> None:
        self.host, self.port = host, port
        self.target = (target_host, target_port or port)
        self._listener: socket.socket | None = None
        self._open: set[socket.socket] = set()
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start listening. Raises OSError when the address is not free or not one of this machine."""
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind((self.host, self.port))
            listener.listen(64)
        except OSError:
            listener.close()
            raise
        self._listener = listener
        threading.Thread(target=self._accept, args=(listener,), name="relay", daemon=True).start()

    def stop(self) -> None:
        with self._lock:
            listener, self._listener = self._listener, None
            connections, self._open = list(self._open), set()
        for sock in ([listener] if listener else []) + connections:
            # shutdown first: on Linux a close alone neither wakes a thread that waits in accept or recv on the
            # socket nor tells the other side that the connection is over.
            for end in (lambda: sock.shutdown(socket.SHUT_RDWR), sock.close):
                try:
                    end()
                except OSError:
                    pass

    @property
    def running(self) -> bool:
        return self._listener is not None

    def _accept(self, listener: socket.socket) -> None:
        while True:
            try:
                client, peer = listener.accept()
            except OSError:
                return  # stopped
            threading.Thread(target=self._serve, args=(client, peer[0]), name="relay-connection", daemon=True).start()

    def _serve(self, client: socket.socket, peer: str) -> None:
        # A listener on 0.0.0.0 also covers 127.0.0.1 whenever WSL does not hold that address itself (the model
        # is not up). Then the relay's own connection to the target would come back to the relay, without end.
        # Connections from this machine are therefore never passed on; local clients use 127.0.0.1 directly.
        if self._own_machine(peer):
            client.close()
            return
        try:
            upstream = socket.create_connection(self.target, timeout=CONNECT_TIMEOUT)
        except OSError:
            client.close()
            return
        upstream.settimeout(None)
        for sock in (client, upstream):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # streamed tokens must not wait for more
        with self._lock:
            if self._listener is None:  # stopped in the meantime
                client.close()
                upstream.close()
                return
            self._open.update((client, upstream))
        back = threading.Thread(target=self._pump, args=(upstream, client), name="relay-back", daemon=True)
        back.start()
        self._pump(client, upstream)
        back.join()
        with self._lock:
            self._open.difference_update((client, upstream))
        for sock in (client, upstream):
            try:
                sock.close()
            except OSError:
                pass

    @staticmethod
    def _own_machine(peer: str) -> bool:
        return ipaddress.ip_address(peer).is_loopback

    @staticmethod
    def _pump(source: socket.socket, sink: socket.socket) -> None:
        try:
            while True:
                data = source.recv(CHUNK)
                if not data:
                    break
                sink.sendall(data)
        except OSError:
            pass
        # This direction is over. Pass that on, so a client that went away ends its request at the server,
        # and a response still on its way the other direction arrives.
        try:
            sink.shutdown(socket.SHUT_WR)
        except OSError:
            pass
