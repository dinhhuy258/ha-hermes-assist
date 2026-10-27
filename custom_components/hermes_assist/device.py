"""The device of one conversation agent, shared by its entities."""

from homeassistant.config_entries import ConfigSubentry
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo

from .const import DOMAIN


def agent_device_info(subentry: ConfigSubentry) -> DeviceInfo:
    """Return the device of an agent subentry."""
    return DeviceInfo(
        identifiers={(DOMAIN, subentry.subentry_id)},
        name=subentry.title,
        manufacturer="Nous Research",
        model="Hermes Agent",
        entry_type=DeviceEntryType.SERVICE,
    )
