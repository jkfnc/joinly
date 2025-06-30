import asyncio
import contextlib
import datetime
import logging
import re
import unicodedata

from fastmcp import Client
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ResourceUpdatedNotification, ServerNotification
from pydantic import AnyUrl

from joinly.server import mcp
from joinly.settings import get_settings
from joinly.types import Transcript

logger = logging.getLogger(__name__)


def transcript_to_messages(transcript: Transcript) -> list[HumanMessage]:
    """Convert a transcript to a list of HumanMessage.

    Args:
        transcript: The transcript to convert.

    Returns:
        A list of HumanMessage objects representing the transcript segments.
    """

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
    """Get a new transcript including only segments starting after given time.

    Args:
        transcript: The original transcript.
        after: The time (seconds) after which to include segments.

    Returns:
        A new Transcript object containing only segments that start
            after the specified time.
    """
    segments = [s for s in transcript.segments if s.start > after]
    return Transcript(segments=segments)


def normalize(s: str) -> str:
    """Normalize a string.

    Args:
        s: The string to normalize.

    Returns:
        The normalized string.
    """
    normalized = unicodedata.normalize("NFKD", s.casefold().strip())
    chars = (c for c in normalized if unicodedata.category(c) != "Mn")
    return re.sub(r"[^\w\s]", "", "".join(chars))


def name_in_transcript(transcript: Transcript, name: str) -> bool:
    """Check if the name is mentioned in the transcript.

    Args:
        transcript: The transcript to check.
        name: The name to look for.

    Returns:
        True if the name is mentioned in the transcript, False otherwise.
    """
    pattern = rf"\b{re.escape(normalize(name))}\b"
    return bool(re.search(pattern, normalize(transcript.text)))


async def run(
    *,
    meeting_url: str | None = None,
    model_name: str = "gpt-4o",
    model_provider: str | None = None,
    name_trigger: bool = False,
) -> None:
    """Simple conversational agent for a meeting.

    Args:
        meeting_url: The URL of the meeting to join.
        model_name: The model to use for the agent.
        model_provider: The provider for the model.
        name_trigger: If True, the agent will only respond if its name is mentioned.
    """
    settings = get_settings()
    transcript_url = AnyUrl("transcript://live")
    transcript_event = asyncio.Event()

    async def _message_handler(message) -> None:  # noqa: ANN001
        if (
            isinstance(message, ServerNotification)
            and isinstance(message.root, ResourceUpdatedNotification)
            and message.root.params.uri == transcript_url
        ):
            logger.info("Transcription update received")
            transcript_event.set()

    llm = init_chat_model(model_name, model_provider=model_provider)

    # More explicit system prompt that forces tool usage
    system_prompt = (
        f"Today is {datetime.datetime.now(tz=datetime.UTC).strftime('%d.%m.%Y')}. "
        f"You are {settings.name}, a voice assistant in a meeting. "
        "CRITICAL RULES - YOU MUST FOLLOW THESE:\n"
        "1. You are in a VOICE meeting - you can ONLY communicate by speaking\n"
        "2. ALWAYS use 'speak_text' tool to say anything - NEVER respond with text alone\n"
        "3. ALWAYS call 'finish' tool after using speak_text\n"
        "4. For unmute requests: use unmute_yourself THEN speak_text THEN finish\n"
        "5. For mute requests: use speak_text THEN mute_yourself THEN finish\n\n"
        "Example response pattern:\n"
        "- User says hello → speak_text('Hello! How can I help you?') → finish()\n"
        "- User says unmute → unmute_yourself() → speak_text('I have unmuted myself') → finish()\n\n"
        "Remember: You CANNOT respond without using tools. Use speak_text for ALL responses."
    )

    client = Client(mcp, message_handler=_message_handler)

    async with client:
        await client.session.subscribe_resource(transcript_url)

        @tool
        def finish() -> str:
            """Finish the conversation turn. MUST be called after speak_text."""
            return "Turn complete"

        tools = await load_mcp_tools(client.session)
        tools.append(finish)
        
        # Force tool use by setting tool_choice
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create tool-calling agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,  # Enable verbose to see what's happening
            handle_parsing_errors=True,
            max_iterations=5,
        )
        
        last_time = -1.0
        recent_context = []
        max_context_items = 10

        await client.call_tool(
            "join_meeting",
            {"meeting_url": meeting_url, "participant_name": settings.name},
        )

        try:
            while True:
                await transcript_event.wait()
                transcript_full = Transcript.model_validate_json(
                    (await client.read_resource(transcript_url))[0].text  # type: ignore[attr-defined]
                )
                transcript = transcript_after(transcript_full, after=last_time)
                transcript_event.clear()
                if not transcript.segments:
                    logger.warning("No new segments in the transcript after update")
                    continue

                if name_trigger and not name_in_transcript(transcript, settings.name):
                    continue

                last_time = transcript.segments[-1].start
                
                # Log transcript segments
                transcript_texts = []
                for segment in transcript.segments:
                    logger.info(
                        '%s: "%s"',
                        segment.speaker if segment.speaker else "User",
                        segment.text,
                    )
                    transcript_texts.append(f"{segment.speaker or 'User'}: {segment.text}")
                
                # Add to context
                recent_context.extend(transcript_texts)
                if len(recent_context) > max_context_items:
                    recent_context = recent_context[-max_context_items:]
                
                # Create input
                current_input = " ".join(s.text for s in transcript.segments)
                
                # Add explicit instruction to use tools
                enhanced_input = (
                    f"{current_input}\n\n"
                    f"Remember: You MUST use speak_text tool to respond, then finish tool."
                )
                
                logger.info("Agent processing input: %s", current_input)
                
                # Process with agent
                try:
                    result = await agent_executor.ainvoke({"input": enhanced_input})
                    
                    # Log what happened
                    if "output" in result:
                        logger.info("Agent final output: %s", result["output"])
                    
                    # Check if speak_text was used
                    tools_used = []
                    if "intermediate_steps" in result:
                        for action, observation in result["intermediate_steps"]:
                            if hasattr(action, "tool"):
                                tools_used.append(action.tool)
                                logger.info("Tool used: %s", action.tool)
                    
                    # If no speak_text was used, force it
                    if "speak_text" not in tools_used and result.get("output"):
                        logger.warning("Agent didn't use speak_text, forcing speech output")
                        try:
                            await client.call_tool("speak_text", {"text": result["output"]})
                        except Exception as e:
                            logger.error("Failed to force speak_text: %s", e)
                        
                except Exception as e:
                    logger.error("Error processing transcript: %s", e)
                    # Try to speak the error
                    try:
                        await client.call_tool("speak_text", {"text": "I encountered an error processing your request."})
                    except:
                        pass
                finally:
                    # Always call finish
                    try:
                        await client.call_tool("finish")
                    except:
                        pass

        finally:
            with contextlib.suppress(Exception):
                await client.call_tool("leave_meeting")
