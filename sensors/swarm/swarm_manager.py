class SwarmManager:
    """
    Manages a swarm of nodes/agents.
    """

    def __init__(self, node_count=5, communication="MQTT"):
        self.node_count = node_count
        self.communication = communication
        self.nodes = [{"id": i, "status": "idle"} for i in range(node_count)]

    def broadcast(self, message):
        for node in self.nodes:
            node["status"] = f"Received: {message}"

    def status_report(self):
        return {node["id"]: node["status"] for node in self.nodes}
