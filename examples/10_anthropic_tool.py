#!/usr/bin/env python3
"""Anthropic Claude native tool use with semantic-frame.

Demonstrates using semantic-frame as a tool in Claude conversations.
The tool provides efficient data analysis without consuming context.

Install: pip install semantic-frame[anthropic]

Requires: ANTHROPIC_API_KEY environment variable
"""

import os

# Check for API key early
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY environment variable not set")
    print()
    print("Set it with:")
    print("  export ANTHROPIC_API_KEY='your-key-here'")
    exit(1)

import anthropic

from semantic_frame.integrations.anthropic import get_anthropic_tool, handle_tool_call


def main() -> None:
    print("=" * 70)
    print("Anthropic Claude Tool Use with semantic-frame")
    print("=" * 70)
    print()

    # Initialize Anthropic client
    client = anthropic.Anthropic()

    # Get the semantic-frame tool definition
    tool = get_anthropic_tool()
    print("Tool registered:", tool["name"])
    print()

    # Example 1: Simple analysis request
    print("-" * 70)
    print("Example 1: Simple data analysis")
    print("-" * 70)

    messages = [
        {
            "role": "user",
            "content": (
                "I have server CPU usage readings over the past hour: "
                "[45, 48, 52, 55, 78, 82, 95, 88, 72, 65, 58, 52]. "
                "Can you analyze this data?"
            ),
        }
    ]

    # Make the API call with the tool
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=[tool],
        messages=messages,
    )

    # Process the response
    process_response(client, response, messages, tool)

    print()

    # Example 2: Data with anomalies
    print("-" * 70)
    print("Example 2: Detecting anomalies in latency data")
    print("-" * 70)

    messages = [
        {
            "role": "user",
            "content": (
                "Here's our API latency in milliseconds for today: "
                "[12, 15, 11, 14, 13, 250, 12, 15, 11, 14, 13, 12]. "
                "Is there anything concerning?"
            ),
        }
    ]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=[tool],
        messages=messages,
    )

    process_response(client, response, messages, tool)


def process_response(
    client: anthropic.Anthropic,
    response: anthropic.types.Message,
    messages: list,
    tool: dict,
) -> None:
    """Process Claude's response, handling tool use if needed."""

    # Check if Claude wants to use a tool
    if response.stop_reason == "tool_use":
        # Find the tool use block
        for block in response.content:
            if block.type == "tool_use":
                print(f"Claude is using tool: {block.name}")
                print(f"With input: {block.input}")
                print()

                # Handle the tool call
                tool_result = handle_tool_call(block.input)
                print("Tool result:")
                print(tool_result)
                print()

                # Send the result back to Claude
                messages.append({"role": "assistant", "content": response.content})
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": tool_result,
                            }
                        ],
                    }
                )

                # Get Claude's final response
                final_response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    tools=[tool],
                    messages=messages,
                )

                print("Claude's analysis:")
                for block in final_response.content:
                    if hasattr(block, "text"):
                        print(block.text)
                return

    # Direct response (no tool use)
    for block in response.content:
        if hasattr(block, "text"):
            print("Claude's response:")
            print(block.text)


if __name__ == "__main__":
    main()
