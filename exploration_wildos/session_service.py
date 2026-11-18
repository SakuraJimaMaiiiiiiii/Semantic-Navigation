"""Identify the WildOS controller to its dedicated target client."""
from navigation.session_server import SessionServer


class WildOSSessionServer(SessionServer):
    live_map_url = None

    def descriptor(self):
        return dict(super().descriptor(), navigation_backend='wildos_graph_dijkstra',
                    live_map_url=self.live_map_url)

    def status(self, job_id=None):
        return dict(super().status(job_id), navigation_backend='wildos_graph_dijkstra',
                    live_map_url=self.live_map_url)
