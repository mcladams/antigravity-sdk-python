# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Example demonstrating ToolContext injection for stateful tools using Agent API.

This example shows how a tool can use ToolContext to maintain state
(specifically, counting fruits mentioned) across multiple turns in a
conversation,
using the high-level Agent API which handles context wiring automatically.
"""

import asyncio
from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging

from google.antigravity.agent import Agent
from google.antigravity.connections.local.local_connection_config import LocalAgentConfig
from google.antigravity.hooks import policy
from google.antigravity.tools.tool_context import ToolContext




def record_fruit(fruit_name: str, count: int, ctx: ToolContext) -> str:
  """Records the mention of fruits and updates the total count.

  Args:
      fruit_name: The name of the fruit.
      count: The number of fruits mentioned.
      ctx: The tool context (injected).

  Returns:
      A summary of the current count for that fruit.
  """
  logging.info("record_fruit called with %s: %d", fruit_name, count)
  current_counts = ctx.get_state("fruit_counts", {})
  current_counts[fruit_name] = current_counts.get(fruit_name, 0) + count
  ctx.set_state("fruit_counts", current_counts)

  total = current_counts[fruit_name]
  return (
      f"Recorded {count} {fruit_name}(s). Total {fruit_name} count is now"
      f" {total}."
  )


def get_fruit_inventory(ctx: ToolContext) -> str:
  """Retrieves the current inventory of all recorded fruits.

  Args:
      ctx: The tool context (injected).

  Returns:
      A string listing all fruits and their counts.
  """
  logging.info("get_fruit_inventory called")
  current_counts = ctx.get_state("fruit_counts", {})
  if not current_counts:
    return "No fruits recorded yet."

  lines = ["Current Fruit Inventory:"]
  for fruit, count in current_counts.items():
    lines.append(f"- {fruit}: {count}")
  return "\n".join(lines)


async def run():
  """Runs the example."""
  config = LocalAgentConfig(
      tools=[record_fruit, get_fruit_inventory],
      system_instructions=(
          "You are a fruit inventory assistant. Use the tools to record fruits "
          "mentioned by the user and to report the inventory. "
          "If the user mentions a fruit and a quantity, use record_fruit. "
          "If the user asks for the inventory or what you have, use"
          " get_fruit_inventory."
      ),
      policies=[
          policy.allow("*")
      ],  # Auto-approve all tools to avoid hanging in simulation
  )

  logging.info("Starting Agent session...")
  async with Agent(config) as agent:
    print("=== ToolContext Fruit Counter Demo ===")

    turns = [
        "I have 5 apples.",
        "And I just got 3 bananas.",
        "Oh, and another 2 apples.",
        "What is my current inventory?",
    ]

    for user_input in turns:
      print(f"\nUser: {user_input}")
      response = await agent.chat(user_input)
      print(f"Agent: {response.text}")


def main(argv: Sequence[str]) -> None:
  del argv
  logging.set_verbosity(logging.INFO)
  asyncio.run(run())


if __name__ == "__main__":
  app.run(main)
