import asyncio
import bittensor as bt
import torch
import time
from transformers import BertTokenizer, BertModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

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

        # Initialize TinyBERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
        self.model = BertModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
        self.model.eval()  # Set model to evaluation mode

    def forward(self, input_text: str) -> torch.Tensor:
        """
        Processes the input text using TinyBERT and returns the pooled output.
        """
        try:
            # Tokenize the input text
            inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Use the [CLS] token representation as the output
            cls_output = outputs.last_hidden_state[:, 0, :]  # Shape: (1, hidden_size)
            return cls_output
        except Exception as e:
            logging.error(f"Error processing task: {e}")
            return torch.tensor([])

    async def send_task(self, task, target_port):
        """
        Sends a task to another node specified by target_port.
        """
        try:
            target_node = f"localhost:{target_port}"  # Localhost with target port
            await bt.network.send_task(target_node, task)
            logging.info(f"Task sent: {task} to {target_node}")
        except Exception as e:
            logging.error(f"Failed to send task to {target_port}: {e}")

    async def process_tasks(self):
        """
        Processes incoming tasks.
        """
        while True:
            if self.tasks:
                task = self.tasks.pop(0)  # Get the next task
                input_text = task.get('input_text', '')
                if not input_text:
                    logging.warning("Received task with empty input_text.")
                    continue
                output = self.forward(input_text)  # Process the task
                if output.numel() > 0:
                    self.rewards += 1  # Increment reward for each task processed
                    self.tao_balance += 0.1  # Increment TAO balance for each task processed
                    logging.info(f"Processed task: '{input_text}' | Rewards: {self.rewards} | TAO Balance: {self.tao_balance:.2f} TAO")
                else:
                    logging.warning("Processing task returned empty output.")
            await asyncio.sleep(1)  # Check for tasks every second

    async def listen_for_tasks(self):
        """
        Simulates listening for tasks. Replace this with actual task listening logic.
        """
        while True:
            # Simulate listening for tasks
            await asyncio.sleep(5)  # Wait for 5 seconds before adding a new task
            # For demonstration, add a dummy task
            dummy_input = "This is a sample input for TinyBERT."  # Replace with actual input
            self.tasks.append({'input_text': dummy_input})
            logging.info(f"Received new task: '{dummy_input}'")

    async def run(self):
        """
        Runs the neuron by concurrently processing and listening for tasks.
        """
        logging.info(f"{self.wallet.name} is running on port {self.port}")
        # Run the task processing and listening concurrently
        await asyncio.gather(self.process_tasks(), self.listen_for_tasks())

# Main logic for running the node
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python node.py <wallet_name> <port>")
        sys.exit(1)

    wallet_name = sys.argv[1]
    port = int(sys.argv[2])
    neuron = MyNeuron(wallet_name, port)

    # Start the event loop
    try:
        asyncio.run(neuron.run())
    except KeyboardInterrupt:
        logging.info("Neuron has been stopped manually.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
