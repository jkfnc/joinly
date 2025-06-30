# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastmcp",
#     "langchain",
#     "langchain-anthropic",
#     "langchain-mcp-adapters",
#     "langchain-ollama",
#     "langchain-openai",
#     "langgraph",
#     "py-dotenv",
#     "rich",
# ]
# ///

import asyncio
import contextlib
import datetime
import json
import logging
import os
import re

from fastmcp import Client
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.memory import ConversationSummaryBufferMemory
from mcp import ResourceUpdatedNotification, ServerNotification
from pydantic import AnyUrl, BaseModel

logger = logging.getLogger(__name__)


class TranscriptSegment(BaseModel):
    """A segment of a transcript."""

    text: str
    start: float
    end: float
    speaker: str | None = None


class Transcript(BaseModel):
    """A transcript containing multiple segments."""

    segments: list[TranscriptSegment]


def transcript_to_messages(transcript: Transcript) -> list[HumanMessage]:
    """Convert a transcript to a list of HumanMessage."""

    def _normalize_speaker(speaker: str | None) -> str:
        if speaker is None:
            return "Unknown"
        speaker = re.sub(r"\s+", "_", speaker.strip())
        return re.sub(r"[<>\|\\\/]+", "", speaker)

    return [
        HumanMessage(
            content=s.text,
            name=_normalize_speaker(s.speaker),
        )
        for s in transcript.segments
    ]


def transcript_after(transcript: Transcript, after: float) -> Transcript:
    """Get a new transcript including only segments starting after given time."""
    segments = [s for s in transcript.segments if s.start > after]
    return Transcript(segments=segments)


def log_agent_output(result: dict) -> None:
    """Log the agent's output and any tool calls made."""
    # Log the final output
    if "output" in result:
        logger.info("Agent response: %s", result["output"])
    
    # Log any intermediate steps (tool calls)
    if "intermediate_steps" in result:
        for action, observation in result["intermediate_steps"]:
            if hasattr(action, "tool"):
                args_str = ", ".join(
                    f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                    for k, v in action.tool_input.items()
                )
                logger.info("%s: %s", action.tool, args_str)
                logger.info("Tool result: %s", observation)


async def run(  # noqa: C901, PLR0915
    mcp_url: str,
    meeting_url: str,
    model_name: str,
    model_provider: str | None = None,
    config: dict | None = None,
) -> None:
    """Simple conversational agent for a meeting using tool-calling agent.

    Args:
        mcp_url: The URL of the MCP server.
        meeting_url: The URL of the meeting to join.
        model_name: The model to use for the agent.
        model_provider: The provider for the model.
        config: Optional configuration for additional MCP servers.
    """
    transcript_url = AnyUrl("transcript://live")
    transcript_event = asyncio.Event()

    async def _message_handler(message) -> None:  # noqa: ANN001
        if (
            isinstance(message, ServerNotification)
            and isinstance(message.root, ResourceUpdatedNotification)
            and message.root.params.uri == transcript_url
        ):
            transcript_event.set()

    llm = init_chat_model(model_name, model_provider=model_provider)

    system_prompt = (
        f"Today is {datetime.datetime.now(tz=datetime.UTC).strftime('%d.%m.%Y')}. "
        "You are joinly, a professional and knowledgeable meeting assistant. "
        "Provide concise, valuable contributions in the meeting. "
        "You are only with one other participant in the meeting, therefore "
        "respond to all messages and questions. "
        "When you are greeted, respond politely in spoken language. "
        "Give information, answer questions, and fullfill tasks as needed. "
        "You receive real-time transcripts from the ongoing meeting. "
        "Respond interactively and use available tools to assist participants. "
        "Always finish your response with the 'finish' tool. "
        "Never directly use the 'finish' tool, always respond first and then use it. "
        "If interrupted mid-response, use 'finish'."
    )

    # separate joinly client, since fastmcp does not support notifications
    # in proxy server mode yet (v2.7.0)
    joinly_client = Client(mcp_url, message_handler=_message_handler)
    client = Client(config) if config and config.get("mcpServers") else None

    mcp_servers = list(config.get("mcpServers", {}).keys()) if config else None
    logger.info(
        "Connecting to joinly MCP server at %s and following other MCP servers: %s",
        mcp_url,
        mcp_servers,
    )
    async with joinly_client, client or contextlib.nullcontext():
        if joinly_client.is_connected():
            logger.info("Connected to joinly MCP server")
        else:
            logger.error("Failed to connect to joinly MCP server at %s", mcp_url)
        if client and not client.is_connected():
            logger.error("Failed to connect to additional MCP servers: %s", mcp_servers)

        await joinly_client.session.subscribe_resource(transcript_url)

        @tool(return_direct=True)
        def finish() -> str:
            """Finish tool to end the turn."""
            return "Finished."

        # load tools from joinly and other MCP servers
        tools = await load_mcp_tools(joinly_client.session)
        if client:
            tools.extend(await load_mcp_tools(client.session))
        tools.append(finish)

        # Create memory for conversation history
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=2000,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create tool-calling agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        # Create agent executor with memory
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,
        )
        
        last_time = -1.0

        logger.info("Joining meeting at %s", meeting_url)
        await joinly_client.call_tool(
            "join_meeting", {"meeting_url": meeting_url, "participant_name": "joinly"}
        )
        logger.info("Joined meeting successfully")

        try:
            while True:
                await transcript_event.wait()
                transcript_full = Transcript.model_validate_json(
                    (await joinly_client.read_resource(transcript_url))[0].text  # type: ignore[attr-defined]
                )
                transcript = transcript_after(transcript_full, after=last_time)
                transcript_event.clear()
                if not transcript.segments:
                    logger.warning("No new segments in the transcript after update")
                    continue

                last_time = transcript.segments[-1].end
                
                # Log transcript segments
                for segment in transcript.segments:
                    logger.info(
                        '%s: "%s"',
                        segment.speaker if segment.speaker else "User",
                        segment.text,
                    )
                
                # Combine all transcript segments into a single input
                transcript_text = " ".join(s.text for s in transcript.segments)
                
                # Process with agent
                try:
                    result = await agent_executor.ainvoke({"input": transcript_text})
                    log_agent_output(result)
                except Exception as e:
                    logger.error("Error processing transcript: %s", e)
                    # Call finish tool to end the turn even on error
                    try:
                        await joinly_client.call_tool("finish")
                    except:
                        pass

        finally:
            logger.info("Leaving meeting")
            with contextlib.suppress(Exception):
                await joinly_client.call_tool("leave_meeting")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from dotenv import load_dotenv
    from rich.logging import RichHandler

    load_dotenv()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description=(
            "Run a conversational agent for a meeting using joinly.ai. "
            "Optionally, connect to different MCP servers."
        )
    )
    parser.add_argument("meeting_url", help="The URL of the meeting to join")
    parser.add_argument(
        "--mcp-url",
        dest="mcp_url",
        default="http://localhost:8000/mcp/",
        help="The URL of the joinly MCP server",
    )
    parser.add_argument(
        "--model-name",
        dest="model_name",
        default=os.getenv("JOINLY_MODEL_NAME", "gpt-4o"),
        help="The model to use for the agent",
    )
    parser.add_argument(
        "--model-provider",
        dest="model_provider",
        default=os.getenv("JOINLY_MODEL_PROVIDER"),
        help="The provider for the model",
    )
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        default=None,
        help=(
            "Path to a JSON configuration file for additional MCP servers. "
            "The file should contain configuration like: "
            '\'{"mcpServers": {"remote": {"url": "https://example.com/mcp"}}}\'. '
            "See https://gofastmcp.com/clients/client for more details."
        ),
    )
    args = parser.parse_args()
    config = None
    if args.config:
        try:
            with Path(args.config).open("r") as f:
                config = json.load(f)
        except Exception:
            logger.exception("Failed to load configuration file")
            args.config = None

    asyncio.run(
        run(
            mcp_url=args.mcp_url,
            meeting_url=args.meeting_url,
            model_name=args.model_name,
            model_provider=args.model_provider,
            config=config,
        )
    )
