import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Self

logger = logging.getLogger(__name__)

_VNC_PORT_RE = re.compile(r"PORT=(\d+)")


class VirtualDisplay:
    """A class to create and dispose an Xvfb display."""

    def __init__(
        self,
        *,
        env: dict[str, str] | None = None,
        size: tuple[int, int] = (1280, 720),
        depth: int = 24,
        use_vnc_server: bool = False,
        vnc_port: int | None = None,
    ) -> None:
        """Initialize the VirtualDisplay.

        Args:
            env: Optional environment dictionary to set the display name.
            size: The display width and height in pixels (default is 1280x720).
            depth: The color depth of the display (default is 24).
            use_vnc_server: Whether to use a VNC server (default is False).
            vnc_port: The port for the VNC server (default is None).
        """
        self._env: dict[str, str] = env if env is not None else {}
        self.size = size
        self.depth = depth
        self.display_name: str | None = None
        self.use_vnc_server = use_vnc_server
        self.vnc_port = vnc_port
        self._vnc_port = vnc_port
        self._proc: asyncio.subprocess.Process | None = None
        self._vnc_proc: asyncio.subprocess.Process | None = None
        self._xvfb_auth_file: tempfile.NamedTemporaryFile | None = None

    async def __aenter__(self) -> Self:
        """Start the Xvfb display."""
        if self._proc is not None:
            msg = "Xvfb already started"
            raise RuntimeError(msg)

        logger.info("Starting Xvfb display")

        self._env["XDG_SESSION_TYPE"] = "x11"
        self._env.pop("WAYLAND_DISPLAY", None)
        
        # Create auth file for Xvfb (helps with WSL)
        self._xvfb_auth_file = tempfile.NamedTemporaryFile(delete=False, prefix="xvfb_auth_")
        auth_file = self._xvfb_auth_file.name

        # WSL-specific: Create X11 unix socket directory if it doesn't exist
        x11_unix_dir = Path("/tmp/.X11-unix")
        if not x11_unix_dir.exists():
            try:
                x11_unix_dir.mkdir(mode=0o1777)
                logger.info("Created /tmp/.X11-unix directory")
            except Exception as e:
                logger.warning(f"Could not create /tmp/.X11-unix: {e}")

        # Try to find an available display number
        display_num = None
        for i in range(99, 110):
            if not (x11_unix_dir / f"X{i}").exists():
                display_num = i
                break
        
        if display_num is None:
            display_num = 99  # fallback

        self.display_name = f":{display_num}"
        logger.info(f"Attempting to start Xvfb on display {self.display_name}")

        # Start Xvfb with explicit display number (more reliable in WSL)
        cmd = [
            "/usr/bin/Xvfb",
            self.display_name,
            "-screen", "0", f"{self.size[0]}x{self.size[1]}x{self.depth}",
            "-nolisten", "tcp",
            "-nolisten", "unix",  # WSL sometimes has issues with unix sockets
            "-auth", auth_file,
            "-ac",  # disable access control (helps with permission issues)
        ]

        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
            start_new_session=True,
        )
        
        # Give Xvfb time to start
        await asyncio.sleep(1.0)
        
        # Check if process is still running
        if self._proc.returncode is not None:
            stderr_data = await self._proc.stderr.read() if self._proc.stderr else b""
            msg = f"Xvfb failed to start. Exit code: {self._proc.returncode}. Stderr: {stderr_data.decode()}"
            logger.error(msg)
            raise RuntimeError(msg)
        
        # Set environment variable
        self._env["DISPLAY"] = self.display_name
        self._env["XAUTHORITY"] = auth_file

        logger.info(
            "Started Xvfb display: %s (size: %dx%d, depth: %d)",
            self.display_name,
            self.size[0],
            self.size[1],
            self.depth,
        )

        # For WSL, we might not have the socket file, but Xvfb should still work
        socket_path = x11_unix_dir / f"X{display_num}"
        if socket_path.exists():
            logger.info(f"X11 socket created at {socket_path}")
        else:
            logger.warning(f"X11 socket not found at {socket_path}, but Xvfb process is running (common in WSL)")

        if self.use_vnc_server:
            cmd = [
                "/usr/bin/x11vnc",
                "-display", self.display_name,
                "-forever",
                "-nopw",
                "-shared",
            ]
            if self.vnc_port is not None:
                cmd.extend(["-rfbport", str(self.vnc_port)])

            self._vnc_proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env,
                start_new_session=True,
            )
            if self._vnc_port is None:
                while line := await self._vnc_proc.stdout.readline():  # type: ignore[attr-defined]
                    m = _VNC_PORT_RE.search(line.decode())
                    if m:
                        self._vnc_port = int(m.group(1))
                        break
                else:
                    logger.warning(
                        "Could not find VNC port in stdout, terminating VNC server"
                    )
                    self._vnc_proc.terminate()

            if self._vnc_port is not None:
                logger.info("Started VNC server on port: %s", self._vnc_port)

        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Stop the Xvfb display."""
        if self._proc is None:
            logger.warning("Xvfb is not started, skipping exit")
            return

        logger.info("Stopping Xvfb display: %s", self.display_name)

        if self._proc.returncode is None:
            self._proc.terminate()
        try:
            await asyncio.wait_for(self._proc.wait(), 5)
        except TimeoutError:
            logger.warning(
                "Xvfb display %s did not stop in time, killing it", self.display_name
            )
            self._proc.kill()
            await self._proc.wait()
        self._proc = None

        if self._env.get("DISPLAY") == self.display_name:
            self._env.pop("DISPLAY", None)
        self._env.pop("XAUTHORITY", None)

        # Clean up auth file
        if self._xvfb_auth_file:
            try:
                os.unlink(self._xvfb_auth_file.name)
            except:
                pass
            self._xvfb_auth_file = None

        logger.info("Stopped Xvfb display: %s", self.display_name)
        self.display_name = None

        if self._vnc_proc is not None:
            logger.info("Stopping VNC server on port: %s", self._vnc_port)
            if self._vnc_proc.returncode is None:
                self._vnc_proc.terminate()
            try:
                await asyncio.wait_for(self._vnc_proc.wait(), 5)
            except TimeoutError:
                logger.warning(
                    "VNC server on port %s did not stop in time, killing it",
                    self._vnc_port,
                )
                self._vnc_proc.kill()
                await self._vnc_proc.wait()
            logger.info("Stopped VNC server on port: %s", self._vnc_port)
            self._vnc_proc = None
            self._vnc_port = None
