import logging
import math
import threading
import time
from typing import Any, List, Optional

import can

from i2rt.motor_drivers.utils import ReceiveMode

ETHERCAT_CHANNEL_PREFIX = "ecat:"
GS_USB_CHANNEL_PREFIX = "gsusb:"


def gs_usb_bus_kwargs(device: str) -> dict[str, Any]:
    """python-can ``gs_usb`` Bus arguments selecting ``device``: an adapter serial, or a scan index.

    gs_usb drives a candleLight-firmware adapter over libusb, with no SocketCAN (i.e. on macOS). The serial
    is the stable choice; the index (``0``, ``1``, ...) is the adapter's position in ``GsUsb.scan()``, which
    can change between runs.
    """
    if device.isdigit():
        return {"channel": int(device)}
    try:
        from gs_usb.gs_usb import GsUsb
    except ImportError as e:
        raise can.CanInterfaceNotImplementedError(
            "the gs_usb interface needs: uv pip install gs_usb pyusb (and brew install libusb)"
        ) from e
    found = {dev.gs_usb.serial_number: dev.gs_usb for dev in GsUsb.scan()}
    if device not in found:
        raise can.CanInitializationError(
            f"no gs_usb adapter with serial {device!r}; connected: {sorted(found) or 'none'}. Pass one of "
            "those serials, or an index (0, 1, ...)."
        )
    # python-can opens a specific gs_usb device by USB bus + address rather than by serial.
    return {"channel": device, "bus": found[device].bus, "address": found[device].address}


def open_can_bus(channel: str, bustype: str = "socketcan", bitrate: int = 1000000) -> can.BusABC:
    """A python-can bus on ``channel``.

    ``ecat:<nic>`` (or ``ecat:<nic>:<slave>:<can channel>``) opens one through an EtherCAT-to-CAN
    board on that NIC instead of a CAN adapter, so every CAN tool that takes a channel works there.
    ``gsusb:<serial>`` (or ``gsusb:<index>``) does the same through a gs_usb adapter without SocketCAN,
    e.g. on macOS.
    """
    if channel.startswith(ETHERCAT_CHANNEL_PREFIX):
        from i2rt.motor_drivers.ethercat_chain import EtherCATCanBus

        return EtherCATCanBus(channel)
    if channel.startswith(GS_USB_CHANNEL_PREFIX):
        device = channel[len(GS_USB_CHANNEL_PREFIX) :]
        return _open_gs_usb_bus(bitrate, **gs_usb_bus_kwargs(device))
    return can.interface.Bus(bustype=bustype, channel=channel, bitrate=bitrate)


# A gs_usb adapter does not survive being closed and reopened in one process (macOS, bench
# 2026-11-12). A plain reopen hangs in libusb; releasing the USB handle and pausing first gets the
# reopen through, but the reopened adapter still loses replies -- a robot start (type check, config
# check, then the chain: three opens) failed to enable a different motor each run, while a single
# open answered 2100/2100 register reads. So the adapter is opened once per process and shared: a
# caller's shutdown() only hands it back, and the USB handle is released when the process exits.
_gs_usb_buses: dict[tuple, can.BusABC] = {}


def gs_usb_read_timeout(timeout: Optional[float]) -> Optional[float]:
    """A recv timeout gs_usb cannot turn into "block forever".

    python-can's gs_usb reader passes ``round(timeout * 1000)`` ms to libusb, and libusb reads 0 ms
    as no timeout at all. So any wait under 0.5 ms -- the tail of a ``deadline - now`` loop, e.g. the
    teaching handle's 1 s version probe -- blocked inside libusb, where Ctrl+C cannot reach it. Round
    such waits up to whole milliseconds instead. None and 0 already map to python-can's 1 ms poll.
    """
    if timeout is None or timeout <= 0:
        return timeout
    return max(0.001, math.ceil(timeout * 1000) / 1000)


_GS_USB_DRIVER_SETTLE_S = 1.0
_gs_usb_start_lock = threading.Lock()


def _gs_usb_start_without_root(self: Any, flags: Optional[int] = None) -> None:
    """``GsUsb.start`` without the step that needs root on macOS; otherwise the same.

    gs_usb's start USB-resets the adapter (the clean start that replaces unplugging it) and then
    detaches any kernel driver it finds. A macOS reset re-enumerates the device, which for a moment
    reports a kernel driver -- gone within 50 ms (bench, 2026-11-23) -- and detaching it is what
    required sudo ("USBError 13 Access denied"). So on macOS this waits that driver out instead.
    Linux keeps the detach: its gs_usb kernel driver is real and stays.
    """
    import platform

    import usb.core
    from gs_usb.constants import (
        GS_CAN_MODE_HW_TIMESTAMP,
        GS_CAN_MODE_LISTEN_ONLY,
        GS_CAN_MODE_LOOP_BACK,
        GS_CAN_MODE_NORMAL,
        GS_CAN_MODE_ONE_SHOT,
    )
    from gs_usb.gs_usb import _GS_USB_BREQ_MODE, GS_CAN_MODE_START
    from gs_usb.gs_usb_structures import DeviceMode

    if flags is None:
        flags = GS_CAN_MODE_NORMAL | GS_CAN_MODE_HW_TIMESTAMP
    device = self.gs_usb
    device.reset()
    system = platform.system()
    if system == "Darwin":
        deadline = time.monotonic() + _GS_USB_DRIVER_SETTLE_S
        while True:
            try:
                active = device.is_kernel_driver_active(0)
            except usb.core.USBError:
                active = True  # still re-enumerating
            if not active:
                break
            if time.monotonic() > deadline:
                device.detach_kernel_driver(0)  # a driver that stayed: needs root, as before
                break
            time.sleep(0.01)
    elif system != "Windows" and device.is_kernel_driver_active(0):
        device.detach_kernel_driver(0)

    flags &= self.device_capability.feature
    flags &= GS_CAN_MODE_LISTEN_ONLY | GS_CAN_MODE_LOOP_BACK | GS_CAN_MODE_ONE_SHOT | GS_CAN_MODE_HW_TIMESTAMP
    self.device_flags = flags
    device.ctrl_transfer(0x41, _GS_USB_BREQ_MODE, 0, 0, DeviceMode(GS_CAN_MODE_START, flags).pack())


def _open_gs_usb_bus(bitrate: int, **kwargs: Any) -> can.BusABC:
    """The process's one python-can gs_usb bus for this adapter, opened on first use."""
    key = (bitrate, *sorted(kwargs.items()))
    bus = _gs_usb_buses.get(key)
    if bus is not None:
        # Hand-over between users (e.g. startup checks -> chain): clear any receive filter the last one
        # set, and drop frames meant for it.
        bus.set_filters(None)
        while bus.recv(timeout=0.005) is not None:
            pass
        return bus

    import atexit

    import usb.core
    import usb.util
    from can.interfaces.gs_usb import GsUsbBus
    from gs_usb.gs_usb_frame import (
        GS_USB_FRAME_SIZE,
        GS_USB_FRAME_SIZE_HW_TIMESTAMP,
        GS_USB_NONE_ECHO_ID,
        GsUsbFrame,
    )

    class _SharedGsUsbBus(GsUsbBus):
        def shutdown(self) -> None:
            """Kept open for the next user in this process; really closed at exit.

            Clears the caller's receive filters: on SocketCAN each open is its own socket, so a filter
            dies with it, but here every user shares this one bus. The teaching-handle check filters to
            encoder ids only, and left in place that hid every motor reply from the chain -- whose bus
            was opened first, so a reset on the next open alone would not reach it.
            """
            self.set_filters(None)

        def _recv_internal(self, timeout: Optional[float]) -> Any:
            """python-can's read, but each frame decoded by its own length.

            gs_usb frames are 24 bytes with a hardware timestamp and 20 without. python-can unpacks
            every frame at the size the device was opened with, so a 20-byte frame -- which this
            adapter sends right after another process has used it (bench, 2026-11-23) -- raised
            "unpack requires a buffer of 24 bytes" and killed whichever script opened it next.
            """
            timeout = gs_usb_read_timeout(timeout)
            timeout_ms = round(timeout * 1000) if timeout else 1
            try:
                data = self.gs_usb.gs_usb.read(0x81, GS_USB_FRAME_SIZE_HW_TIMESTAMP, timeout_ms)
            except usb.core.USBError:
                return None, False
            if len(data) not in (GS_USB_FRAME_SIZE, GS_USB_FRAME_SIZE_HW_TIMESTAMP):
                logging.debug("gs_usb: dropped a %d-byte USB packet", len(data))
                return None, False
            frame = GsUsbFrame()
            GsUsbFrame.unpack_into(frame, bytes(data), len(data) == GS_USB_FRAME_SIZE_HW_TIMESTAMP)
            msg = can.Message(
                timestamp=frame.timestamp,
                arbitration_id=frame.arbitration_id,
                is_extended_id=frame.is_extended_id,
                is_remote_frame=frame.is_remote_frame,
                is_error_frame=frame.is_error_frame,
                channel=self.channel_info,
                dlc=frame.can_dlc,
                data=bytearray(frame.data)[0 : frame.can_dlc],
                is_rx=frame.echo_id == GS_USB_NONE_ECHO_ID,
            )
            return msg, False

        def _close(self) -> None:
            if self._is_shutdown:
                return
            try:
                GsUsbBus.shutdown(self)
            finally:
                usb.util.dispose_resources(self.gs_usb.gs_usb)

    # python-can's GsUsbBus calls GsUsb.start() on a device it creates itself, so the root-free start
    # is swapped in for just this construction.
    from gs_usb.gs_usb import GsUsb

    with _gs_usb_start_lock:
        original_start = GsUsb.start
        GsUsb.start = _gs_usb_start_without_root
        try:
            bus = _SharedGsUsbBus(bitrate=bitrate, **kwargs)
        finally:
            GsUsb.start = original_start
    _gs_usb_buses[key] = bus
    atexit.register(bus._close)
    return bus


def _frame_str(msg: Optional[can.Message]) -> str:
    return "nothing" if msg is None else f"0x{msg.arbitration_id:x} [{bytes(msg.data).hex(' ')}]"


class CanInterface:
    def __init__(
        self,
        channel: str = "PCAN_USBBUS1",
        bustype: str = "socketcan",
        bitrate: int = 1000000,
        name: str = "default_can_interface",
        receive_mode: ReceiveMode = ReceiveMode.p16,
        use_buffered_reader: bool = False,
    ):
        self.channel = channel
        self.bus = open_can_bus(channel, bustype, bitrate)
        self.busstate = self.bus.state
        self.name = name
        self.receive_mode = receive_mode
        self.use_buffered_reader = use_buffered_reader
        logging.info(f"Can interface {self.name} use_buffered_reader: {use_buffered_reader}")
        if use_buffered_reader:
            # Initialize BufferedReader for asynchronous message handling
            self.buffered_reader = can.BufferedReader()
            self.notifier = can.Notifier(self.bus, [self.buffered_reader])

    def close(self) -> None:
        """Shut down the CAN bus."""
        if self.use_buffered_reader:
            self.notifier.stop()
        self.bus.shutdown()

    def _send_message_get_response(
        self, id: int, motor_id: int, data: List[int], max_retry: int = 5, expected_id: Optional[int] = None
    ) -> can.Message:
        """Send a message over the CAN bus.

        Args:
            id (int): The arbitration ID of the message.
            data (List[int]): The data payload of the message.

        Returns:
            can.Message: The message that was sent.
        """
        message = can.Message(arbitration_id=id, data=data, is_extended_id=False)
        attempts: List[str] = []  # what each try got back, for the failure message
        for _ in range(max_retry):
            try:
                # logging.info("Sending message: %s at %f", message, time.time())
                self.bus.send(message)
                if expected_id is None:
                    expected_id = self.receive_mode.get_receive_id(motor_id)
                # Skip other motors' frames until this one answers. Stale replies can be queued ahead of
                # it -- motor_on's clear-error retries leave several behind (bench, gs_usb on macOS) --
                # and taking only the first frame then failed the next motor on every retry.
                skipped: List[can.Message] = []
                deadline = time.time() + 0.01
                while (remaining := deadline - time.time()) > 0:
                    # None on timeout (this class's _receive_message does not raise): retry the send.
                    response = self._receive_message(motor_id, timeout=remaining, supress_warning=True)
                    if response is None:
                        break
                    if response.arbitration_id == expected_id:
                        return response
                    skipped.append(response)
                attempts.append(
                    f"no 0x{expected_id:x} reply; skipped {len(skipped)} other frame(s)"
                    + (f", last {_frame_str(skipped[-1])}" if skipped else "")
                )
            except (can.CanError, AssertionError) as e:
                attempts.append(f"{type(e).__name__}: {e}")
                logging.warning(e)
                logging.warning(
                    "\033[91m"
                    + f"CAN Error {self.name}: Failed to communicate with motor {id} over can bus. Retrying..."
                    + "\033[0m"
                )
            time.sleep(0.001)
        raise AssertionError(
            f"fail to communicate with the motor {id} on {self.name} at can channel {self.bus.channel_info}; "
            + "; ".join(f"try {i + 1}: {a}" for i, a in enumerate(attempts))
        )

    def try_receive_message(self, motor_id: Optional[int] = None, timeout: float = 0.009) -> Optional[can.Message]:
        """Try to receive a message from the CAN bus.

        Args:
            timeout (float): The time to wait for a message (in seconds).

        Returns:
            can.Message: The received message, or None if no message is received.
        """
        try:
            return self._receive_message(motor_id, timeout, supress_warning=True)
        except AssertionError:
            return None

    def _drain_bus(self, timeout_s: float = 0.05, idle_count: int = 10) -> int:
        """Drain pending CAN frames until the bus is idle.

        Loops `try_receive_message(timeout=0.001)` until either `idle_count`
        consecutive 1 ms reads return None or `timeout_s` wall-clock has
        elapsed. Used at init handovers (e.g. between encoder validation and
        motor bring-up) to flush stale frames that would otherwise be misread
        as the next motor's reply. Returns the number of frames consumed.
        """
        drained = 0
        idle = 0
        deadline = time.time() + timeout_s
        while time.time() < deadline and idle < idle_count:
            if self.try_receive_message(timeout=0.001) is None:
                idle += 1
            else:
                idle = 0
                drained += 1
        return drained

    def _receive_message(
        self, motor_id: Optional[int] = None, timeout: float = 0.009, supress_warning: bool = False
    ) -> Optional[can.Message]:
        """Receive a message from the CAN bus.

        Args:
            timeout (float): The time to wait for a message (in seconds).

        Returns:
            can.Message: The received message.

        Raises:
            AssertionError: If no message is received within the timeout.
        """
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            if self.use_buffered_reader:
                message = self.buffered_reader.get_message(timeout=0.001)
            else:
                message = self.bus.recv(timeout=0.001)
            # Skip our own transmit echoes: gs_usb (the macOS path) returns every sent frame marked
            # is_rx=False, which SocketCAN never does. See RawCanInterface._receive_message.
            if message and message.is_rx:
                return message
        if not supress_warning:
            logging.warning(
                "\033[91m"
                + f"Failed to receive message, {self.name} motor id {motor_id} motor timeout. Check if the motor is powered on or if the motor ID exists."
                + "\033[0m"
            )
