import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "aitads_augmented" / "data"
CONFIG_DIR = ROOT / "aitads_augmented" / "configs"
SOURCE_CONFIG = CONFIG_DIR / "power-grid-minimal.json"
TARGET_CONFIG = CONFIG_DIR / "power-grid-full.json"


EVENT_MAP = {
    "service_scan": ("substation_recon", "substation_recon", "0", "DNP3-Recon"),
    "dns_scan": ("substation_recon", "substation_recon", "0", "IEC104-Recon"),
    "wpscan": ("ied_enumeration", "ied_enumeration", "0", "MMS-Enum"),
    "dirb": ("ied_enumeration", "ied_enumeration", "0", "IED-Enum"),
    "webshell_cmd": (
        "control_gateway_compromise",
        "control_gateway_compromise",
        "0",
        "Gateway-Cmd",
    ),
    "crack_passwords": (
        "operator_account_attack",
        "operator_account_attack",
        "0",
        "Op-Brute",
    ),
    "online_cracking": (
        "operator_account_attack",
        "operator_account_attack",
        "0",
        "Op-Crack",
    ),
    "attacker_change_user": (
        "operator_role_switch",
        "operator_role_switch",
        "0",
        "Role-Switch",
    ),
    "escalated_sudo_command": (
        "control_privilege_escalation",
        "control_privilege_escalation",
        "0",
        "Priv-Esc",
    ),
    "dnsteal_start": (
        "telemetry_exfiltration",
        "telemetry_exfiltration_start",
        "start",
        "PMU-Exfil-Start",
    ),
    "dnsteal_active": (
        "telemetry_exfiltration",
        "telemetry_exfiltration_active",
        "active",
        "PMU-Exfil-Active",
    ),
    "dnsteal_end": (
        "telemetry_exfiltration",
        "telemetry_exfiltration_end",
        "end",
        "PMU-Exfil-End",
    ),
    "dnsteal": (
        "telemetry_exfiltration",
        "telemetry_exfiltration_active",
        "active",
        "PMU-Exfil-Active",
    ),
}

ATTACK_NAMES = {
    "substation_recon": "SCADA IDS: DNP3/IEC104 reconnaissance against substation control network",
    "ied_enumeration": "SCADA IDS: IEC 61850 MMS object enumeration on protection IED",
    "control_gateway_compromise": "Wazuh: Unauthorized command execution on control gateway",
    "operator_account_attack": "Wazuh: Operator credential attack on control-center account",
    "operator_role_switch": "Wazuh: Unauthorized operator role switch detected",
    "control_privilege_escalation": "Wazuh: Control host privilege escalation attempt",
    "telemetry_exfiltration_start": "SCADA IDS: PMU telemetry exfiltration session started",
    "telemetry_exfiltration_active": "SCADA IDS: PMU telemetry exfiltration in progress",
    "telemetry_exfiltration_end": "SCADA IDS: PMU telemetry exfiltration session ended",
}

NOISE_ALERTS = [
    ("SCADA: Routine DNP3 polling response from RTU", "DNP3-Poll"),
    ("IEC61850: GOOSE heartbeat retransmission observed", "GOOSE-Hb"),
    ("Historian: PMU telemetry batch committed", "PMU-Batch"),
    ("Relay: Protection IED self-test status notice", "Relay-Test"),
    ("Control Center: Operator workstation health check", "HMI-Health"),
    ("Substation Firewall: Allowed engineering workstation session", "FW-Allow"),
]

ASSETS = [
    "control_center_hmi",
    "control_center_historian",
    "substation_a_rtu",
    "substation_a_ied",
    "substation_a_relay",
    "substation_b_rtu",
    "substation_b_ied",
    "substation_b_relay",
    "pmu_gateway",
    "engineering_workstation",
    "scada_firewall",
    "control_gateway",
]


def stable_index(value: str, modulo: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def power_ip(asset: str) -> str:
    idx = stable_index(asset, 200)
    return f"10.66.{idx // 50}.{idx % 50 + 10}"


def power_asset(alert: dict) -> str:
    event_label = alert["event_label"]
    if event_label in {"webshell_cmd", "escalated_sudo_command"}:
        return "control_gateway"
    if event_label in {"crack_passwords", "online_cracking", "attacker_change_user"}:
        return "control_center_hmi"
    if event_label.startswith("dnsteal"):
        return "pmu_gateway"
    if event_label in {"service_scan", "dns_scan"}:
        return "scada_firewall"
    if event_label in {"wpscan", "dirb"}:
        return "substation_a_ied"
    return ASSETS[stable_index(alert["host"] + alert["ip"], len(ASSETS))]


def dnsteal_stage_from_file(source_name: str) -> str:
    if source_name.endswith("dnsteal_start"):
        return "dnsteal_start"
    if source_name.endswith("dnsteal_end"):
        return "dnsteal_end"
    if source_name.endswith("dnsteal_active"):
        return "dnsteal_active"
    return "dnsteal"


def transform_alert(alert: dict, source_name: str) -> dict:
    transformed = dict(alert)
    event_label = alert["event_label"]

    asset = power_asset(alert)
    transformed["host"] = asset
    transformed["ip"] = power_ip(asset)

    if event_label == "-":
        noise_name, noise_short = NOISE_ALERTS[
            stable_index(alert["short"] + alert["host"], len(NOISE_ALERTS))
        ]
        transformed["name"] = noise_name
        transformed["short"] = noise_short
        transformed["time_label"] = "routine_scada_noise"
        transformed["event_label"] = "-"
        transformed["hierarchical_event_label"] = "-"
        return transformed

    lookup_label = dnsteal_stage_from_file(source_name) if event_label == "dnsteal" else event_label
    mapped_event, mapped_time, mapped_stage, mapped_short = EVENT_MAP[lookup_label]
    transformed["name"] = ATTACK_NAMES[mapped_time]
    transformed["short"] = mapped_short
    transformed["time_label"] = mapped_time
    transformed["event_label"] = mapped_event
    transformed["hierarchical_event_label"] = f"{mapped_event}.{mapped_stage}."
    return transformed


def referenced_files(config: dict) -> list[str]:
    names = []
    for split in ("train", "val", "test"):
        for scenario in config[split]:
            for day in scenario:
                names.extend(day["noise"])
                names.extend(attack[0] for attack in day["attacks"])
    return sorted(set(names))


def power_name(name: str) -> str:
    return f"power-{name}"


def rewrite_config(config: dict) -> dict:
    rewritten = json.loads(json.dumps(config))
    rewritten["name"] = "power-grid-full"
    rewritten["description"] = (
        "Field-level electric power grid reconstruction generated from "
        "power-grid-minimal. This configuration references power-*.json "
        "fragments whose alert fields are rewritten with SCADA, substation, "
        "IED, PMU, and control-center terminology."
    )
    for split in ("train", "val", "test"):
        for scenario in rewritten[split]:
            for day in scenario:
                day["noise"] = [power_name(name) for name in day["noise"]]
                day["attacks"] = [[power_name(name), time] for name, time in day["attacks"]]
    return rewritten


def main() -> None:
    config = json.loads(SOURCE_CONFIG.read_text())
    for name in referenced_files(config):
        source = DATA_DIR / f"{name}.json"
        target = DATA_DIR / f"{power_name(name)}.json"
        alerts = json.loads(source.read_text())
        transformed = [transform_alert(alert, name) for alert in alerts]
        target.write_text(json.dumps(transformed, indent=2) + "\n")

    rewritten = rewrite_config(config)
    TARGET_CONFIG.write_text(json.dumps(rewritten, indent=4) + "\n")


if __name__ == "__main__":
    main()
