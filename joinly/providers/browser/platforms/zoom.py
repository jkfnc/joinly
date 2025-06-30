import asyncio
import contextlib
import logging
import re
from datetime import UTC, datetime
from typing import Any, ClassVar

from playwright.async_api import Page

from joinly.providers.browser.platforms.base import BaseBrowserPlatformController
from joinly.settings import get_settings
from joinly.types import MeetingChatHistory, MeetingChatMessage, MeetingParticipant

logger = logging.getLogger(__name__)

_TIME_RX = re.compile(r"^\d{1,2}:\d{2}(?:\s*[AP]M)?$", re.IGNORECASE)


class ZoomBrowserPlatformController(BaseBrowserPlatformController):
    """Controller for managing Zoom browser meetings."""

    url_pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?:https?://)?(?:[a-z0-9-]+\.)?zoom\.us/"
    )

    def __init__(self) -> None:
        """Initialize the Zoom browser platform controller."""
        self._state: dict[str, Any] = {}

    @property
    def active_speaker(self) -> str | None:
        """Get the name of the active speaker in the Zoom meeting."""
        return self._state.get("active_speaker")

    async def join(
        self,
        page: Page,
        url: str,
        name: str,
        passcode: str | None = None,
    ) -> None:
        """Join the Zoom meeting.

        Args:
            page: The Playwright page instance.
            url: The URL of the Zoom meeting.
            name: The name of the participant.
            passcode: The passcode for the meeting (if required).
        """
        # convert standard join URL to the web client format
        if re.search(r"/j/\d+", url):
            url = re.sub(r"/j/(\d+)", r"/wc/join/\1", url)
            logger.info("Rewrote Zoom join URL to web client format: %s", url)

        await page.goto(url, wait_until="load", timeout=20000)
        if await page.get_by_text("invalid").is_visible(timeout=2000):
            msg = "Meeting link is invalid."
            raise ValueError(msg)

        async def _click_task(page: Page, text: str) -> None:
            await page.locator("button", has_text=text).first.click(timeout=0)

        pre_tasks = [
            asyncio.create_task(_click_task(page, t))
            for t in ["accept cookies", "i agree"]
        ]

        try:
            name_field = page.locator("#input-for-name")
            await name_field.fill(name, timeout=20000)

            passcode_field = page.locator("input[type='password']")
            if await passcode_field.is_visible(timeout=1000):
                if passcode is not None:
                    await passcode_field.fill(passcode, timeout=1000)
                else:
                    msg = "Passcode is required but not provided."
                    raise ValueError(msg)

            join_btn = page.locator("button", has_text="join").first
            await join_btn.click(timeout=1000)

            await page.wait_for_timeout(2000)
            if await join_btn.is_visible(timeout=1000):
                with contextlib.suppress(Exception):
                    await join_btn.click(timeout=1000)
        finally:
            for task in pre_tasks:
                if not task.done():
                    task.cancel()

        if not await self._check_joined(page):
            msg = "Join check failed: Failed to join the Zoom meeting."
            raise RuntimeError(msg)

        await self._setup_active_speaker_observer(page)

    async def leave(self, page: Page) -> None:
        """Leave the Zoom meeting using the icon-based button."""
        await self._activate_controls(page)

        leave_btn = page.get_by_role("button", name=re.compile(r"leave", re.IGNORECASE))
        if not await leave_btn.is_visible(timeout=1000):
            msg = "Leave button not found or not visible."
            raise RuntimeError(msg)
        await leave_btn.click(timeout=1000)

        leave_btn_confirm = page.locator("button", has_text="leave meeting").first
        if not await leave_btn_confirm.is_visible(timeout=1000):
            with contextlib.suppress(Exception):
                await leave_btn.click(timeout=1000)

        if not await leave_btn_confirm.is_visible(timeout=1000):
            msg = "Leave meeting confirmation button not found or not visible."
            raise RuntimeError(msg)
        await leave_btn_confirm.click(timeout=1000)
        await page.wait_for_timeout(500)

    async def send_chat_message(self, page: Page, message: str) -> None:
        """Send a chat message in Zoom."""
        if len(message) > 1024:  # noqa: PLR2004
            msg = "Message exceeds the maximum length of 1024 characters."
            raise ValueError(msg)

        await self._open_chat(page)

        chat_input = page.locator("div[contenteditable='true']")
        if not await chat_input.is_visible(timeout=1000):
            msg = "Chat input not found or not visible."
            raise RuntimeError(msg)
        await chat_input.click(timeout=1000)
        await page.wait_for_timeout(200)
        await chat_input.fill(message)
        await page.wait_for_timeout(500)
        await page.keyboard.press("Enter")

    async def get_chat_history(self, page: Page) -> MeetingChatHistory:
        """Return a Zoom in-meeting chat history.

        Args:
            page: The Playwright page instance.

        Returns:
            MeetingChatHistory: The chat history of the meeting.
        """
        await self._open_chat(page)

        messages: list[MeetingChatMessage] = []

        panel = page.locator('div[role="application"][aria-label="Chat Message List"]')
        rows = await panel.locator('[role="row"][aria-label]').all()

        for row in rows:
            aria = await row.get_attribute("aria-label") or ""
            parts = [p.strip() for p in aria.split(",")]

            sender: str | None = None
            ts: float | None = None

            if parts and len(parts) >= 3:  # noqa: PLR2004
                first = parts[0]
                sender = (
                    (first.split(" to ")[0].strip() or None)
                    if " to " in first
                    else (first or None)
                )

                raw_time = re.sub(r"[\u00A0\u202F]", "", parts[1]).strip()
                if _TIME_RX.fullmatch(raw_time):
                    if raw_time[-2:].upper() in {"AM", "PM"}:
                        fmt = "%I:%M %p" if " " in raw_time else "%I:%M%p"
                    else:
                        fmt = "%H:%M"
                    clean_time = raw_time.upper().strip()
                    t = datetime.strptime(clean_time, fmt).replace(tzinfo=UTC)
                    today = datetime.now(UTC).date()
                    t = t.replace(year=today.year, month=today.month, day=today.day)
                    ts = t.timestamp()

                text = ",".join(parts[2:]).strip()

                if text:
                    messages.append(
                        MeetingChatMessage(text=text, timestamp=ts, sender=sender)
                    )

        return MeetingChatHistory(messages=messages)

    async def get_participants(self, page: Page) -> list[MeetingParticipant]:
        """Get the list of participants in the Zoom meeting.

        Args:
            page: The Playwright page instance.

        Returns:
            list[MeetingParticipant]: A list of participants in the meeting.
        """
        participants_list = page.locator(
            'div[role="list"][aria-label^="participants" i]'
        )
        is_participant_list_visible = await participants_list.is_visible(timeout=1000)

        participants_button = page.get_by_role(
            "button", name=re.compile(r"participants", re.IGNORECASE)
        )
        if not is_participant_list_visible:
            await self._activate_controls(page)
            if not await participants_button.is_visible(timeout=1000):
                msg = "Participants button not found or not visible."
                raise RuntimeError(msg)
            await participants_button.click(timeout=1000)
            if not await participants_list.is_visible(timeout=1000):
                with contextlib.suppress(Exception):
                    await participants_button.click(timeout=1000)
                await page.wait_for_timeout(1000)

        participants: list[MeetingParticipant] = []
        for item in await participants_list.locator("div.participants-li").all():
            if aria_label := await item.get_attribute("aria-label"):
                labels = aria_label.split(",")
                name = labels[0].strip()
                infos = labels[1:] if len(labels) > 1 else []
                participants.append(MeetingParticipant(name=name, infos=infos))

        await participants_button.click(timeout=1000)

        return participants


    async def mute(self, page: Page) -> None:
        """Mute the microphone in Zoom."""
        await self._activate_controls(page)
        
        # Log current button states for debugging
        logger.info("Looking for mute/unmute buttons...")
        
        # Check if we're already muted by looking for unmute button
        unmute_visible = False
        try:
            unmute_btn = page.get_by_role("button", name="unmute my microphone")
            unmute_visible = await unmute_btn.is_visible(timeout=500)
        except:
            pass
            
        if unmute_visible:
            logger.info("Already muted (unmute button is visible)")
            return

        # Try to find and click the mute button
        try:
            mute_btn = page.get_by_role("button", name="mute my microphone")
            if await mute_btn.is_visible(timeout=1000):
                await mute_btn.click(timeout=1000)
                logger.info("Successfully clicked mute button")
                return
        except Exception as e:
            logger.debug(f"Could not find mute button with get_by_role: {e}")
        
        # Try alternative selector
        try:
            mute_alt = page.locator("button[aria-label*='mute my microphone']").first
            if await mute_alt.is_visible(timeout=1000):
                await mute_alt.click(timeout=1000)
                logger.info("Successfully clicked mute button (alt selector)")
                return
        except Exception as e:
            logger.debug(f"Could not find mute button with alt selector: {e}")
        
        # If we still can't find it, log available buttons
        try:
            all_buttons = await page.locator("button").all()
            audio_buttons = []
            for btn in all_buttons[:20]:  # Check first 20 buttons
                aria_label = await btn.get_attribute("aria-label") or ""
                if any(word in aria_label.lower() for word in ["mute", "microphone", "audio", "mic"]):
                    is_visible = await btn.is_visible()
                    audio_buttons.append(f"{aria_label} (visible: {is_visible})")
            logger.error(f"Could not find mute button. Audio-related buttons found: {audio_buttons}")
        except:
            pass
            
        msg = "Mute button not found or not visible."
        raise RuntimeError(msg)

    async def unmute(self, page: Page) -> None:
        """Unmute the microphone in Zoom."""
        await self._activate_controls(page)
        
        # Log current button states for debugging
        logger.info("Looking for unmute/mute buttons...")
        
        # Check if we're already unmuted by looking for mute button
        mute_visible = False
        try:
            mute_btn = page.get_by_role("button", name="mute my microphone")
            mute_visible = await mute_btn.is_visible(timeout=500)
        except:
            pass
            
        if mute_visible:
            logger.info("Already unmuted (mute button is visible)")
            return

        # Try to find and click the unmute button
        try:
            unmute_btn = page.get_by_role("button", name="unmute my microphone")
            if await unmute_btn.is_visible(timeout=1000):
                await unmute_btn.click(timeout=1000)
                logger.info("Successfully clicked unmute button")
                # Sometimes Zoom requires a second click
                await page.wait_for_timeout(500)
                if await unmute_btn.is_visible(timeout=500):
                    logger.info("Clicking unmute again to confirm")
                    await unmute_btn.click(timeout=1000)
                return
        except Exception as e:
            logger.debug(f"Could not find unmute button with get_by_role: {e}")
        
        # Try alternative selector
        try:
            unmute_alt = page.locator("button[aria-label*='unmute my microphone']").first
            if await unmute_alt.is_visible(timeout=1000):
                await unmute_alt.click(timeout=1000)
                logger.info("Successfully clicked unmute button (alt selector)")
                # Check for second click
                await page.wait_for_timeout(500)
                if await unmute_alt.is_visible(timeout=500):
                    logger.info("Clicking unmute again to confirm")
                    await unmute_alt.click(timeout=1000)
                return
        except Exception as e:
            logger.debug(f"Could not find unmute button with alt selector: {e}")
        
        # If we still can't find it, log available buttons
        try:
            all_buttons = await page.locator("button").all()
            audio_buttons = []
            for btn in all_buttons[:20]:  # Check first 20 buttons
                aria_label = await btn.get_attribute("aria-label") or ""
                if any(word in aria_label.lower() for word in ["mute", "microphone", "audio", "mic"]):
                    is_visible = await btn.is_visible()
                    audio_buttons.append(f"{aria_label} (visible: {is_visible})")
            logger.error(f"Could not find unmute button. Audio-related buttons found: {audio_buttons}")
        except:
            pass
            
        msg = "Unmute button not found or not visible."
        raise RuntimeError(msg)

    async def _check_joined(self, page: Page, timeout: float = 10) -> bool:  # noqa: ASYNC109
        """Check if the Zoom meeting has been joined successfully.

        Args:
            page: The Playwright page instance.
            timeout: The timeout in seconds for checking the join status.

        Returns:
            bool: True if joined, False otherwise.
        """
        locators = [
            page.locator("span >> text=/joining/i"),
            page.get_by_role("button", name=re.compile(r"leave", re.IGNORECASE)),
        ]

        tasks = [
            asyncio.create_task(loc.wait_for(state="visible", timeout=0))
            for loc in locators
        ]

        try:
            done, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED, timeout=timeout
            )
            return any(not task.exception() for task in done)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def _activate_controls(self, page: Page) -> None:
        """Activate control bar."""
        await page.mouse.click(640, 360)
        await page.wait_for_timeout(100)

    async def _open_chat(self, page: Page) -> None:
        """Open the chat in the Zoom meeting."""
        chat_input = page.locator("div[contenteditable='true']")
        is_chat_visible = await chat_input.is_visible(timeout=1000)

        if not is_chat_visible:
            await self._activate_controls(page)
            chat_button = page.get_by_role(
                "button", name=re.compile(r"chat panel", re.IGNORECASE)
            )
            if not await chat_button.is_visible(timeout=1000):
                msg = "Chat button not found or not visible."
                raise RuntimeError(msg)
            await chat_button.click(timeout=1000)
            if not await chat_input.is_visible(timeout=1000):
                with contextlib.suppress(Exception):
                    await chat_button.click(timeout=1000)
                await page.wait_for_timeout(1000)

    async def _setup_active_speaker_observer(self, page: Page) -> None:
        """Setup the active speaker observer for Zoom."""
        await page.expose_binding(
            "report",
            lambda _, name: self._state.update({"active_speaker": name}),
        )
        await page.evaluate(
            """
            (nameArg) => {
                const emit = n => window.report(n);
                const find = () => {
                    const selectors = [
                        'div.speaker-active-container__video-frame span',
                        'div.speaker-bar-container__video-frame--active span',
                        'div.speaker-bar-container__video-frame span',
                    ];

                    const name = selectors
                        .flatMap(sel => Array.from(document.querySelectorAll(sel)))
                        .map(el => el?.textContent?.trim())
                        .find(text => text && text.length > 0 && text !== nameArg);
                    return name || null;
                };

                let last = null, cur;
                new MutationObserver(() => {
                    cur = find();
                    if (cur !== last) { last = cur; emit(cur); }
                }).observe(
                    document,
                    {
                        subtree: true,
                        childList: true,
                        attributes: true,
                        attributeFilter: ['class']
                    }
                );
                emit(find());
            }
            """,
            get_settings().name,
        )
