import asyncio
import bittensor as bt
import torch
import time

# Define a neuron class
class MyNeuron(bt.Dendrite):
    def __init__(self, wallet_name, port):
        # Create a wallet for this node.
        self.wallet = bt.Wallet(name=wallet_name)
        # Set up the neuron with the wallet
        super(MyNeuron, self).__init__(wallet=self.wallet)
        # Set port for the neuron
        self.port = port
        self.tasks = []  # To hold incoming tasks
        self.rewards = 0  # Initialize rewards
        self.tao_balance = 0.0  # Simulate TAO balance

    def forward(self, inputs_x: torch.Tensor) -> torch.Tensor:
        # Sample processing of inputs. Replace with your own task.
        output = torch.sigmoid(inputs_x)
        return output
    
    async def send_task(self, task):
        # Method to send a task to another node
        # Replace 'target_node' with the actual target node's address
        target_node = "node_address_here"  # Replace this with actual logic
        await bt.network.send_task(target_node, task)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Task sent: {task} to {target_node}")

    async def process_tasks(self):
        while True:
            if self.tasks:
                task = self.tasks.pop(0)  # Get the next task
                inputs = task['inputs']  # Extract inputs from the task
                output = self.forward(inputs)  # Process the task
                self.rewards += 1  # Increment reward for each task processed
                self.tao_balance += 0.1  # Increment TAO balance for each task processed
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processed task: {task} | "
                      f"Current Rewards: {self.rewards} | TAO Balance: {self.tao_balance:.2f} TAO")
            await asyncio.sleep(1)  # Check for tasks every second

    async def listen_for_tasks(self):
        while True:
            # Simulate listening for tasks (you will implement actual logic here)
            await asyncio.sleep(5)  # Simulate delay for receiving tasks
            # For demonstration, add a dummy task
            dummy_input = torch.tensor([0.5])  # Replace with actual input
            self.tasks.append({'inputs': dummy_input})
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Received new task: {dummy_input}")

    async def run(self):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {self.wallet.name} is running on port {self.port}")
        # Run the task processing and listening concurrently
        await asyncio.gather(self.process_tasks(), self.listen_for_tasks())

# Main logic for running the node
if __name__ == "__main__":
    import sys
    wallet_name = sys.argv[1]
    port = int(sys.argv[2])
    neuron = MyNeuron(wallet_name, port)

    # Start the event loop
    asyncio.run(neuron.run())
