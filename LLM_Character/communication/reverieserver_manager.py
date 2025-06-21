from LLM_Character.depreciated.world.game import ReverieServer


class ReverieServerManager:
    def __init__(self):
        self._connections: dict[str, ReverieServer] = {}

    def add_connection(self, clientid: str, server: ReverieServer):
        if clientid not in self._connections.keys():
            self._connections[clientid] = server

    def remove_connection(self, clientid: str):
        if clientid in self._connections.keys():
            self._connection[clientid] = []

    def get_server(self, clientid: str) -> ReverieServer:
        if clientid in self._connections.keys():
            return self._connections[clientid]
        return None
