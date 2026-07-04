# Power Grid Minimal Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal AIT-ADS-A configuration that reuses existing alert fragments to simulate electric power grid alert grouping conditions.

**Architecture:** The implementation adds one new JSON recipe under `aitads_augmented/configs` and one short documentation file. It does not copy or mutate `aitads_augmented/data/*.json`; the dataset loader continues to use `AITAlertDataset(split=..., configuration="power-grid-minimal")`.

**Tech Stack:** AIT-ADS-A JSON configuration, existing `alertbert.aitads.AITAlertDataset`, Markdown documentation.

---

### Task 1: Add Minimal Power Grid Configuration

**Files:**
- Create: `aitads_augmented/configs/power-grid-minimal.json`

- [ ] **Step 1: Create the configuration**

Add a JSON config that reuses small existing alert fragments from `shaw`, `wardbeck`, `russellmitchell`, and `santos`. The scenario comments map existing AIT-ADS attack fragments onto power-grid-inspired meanings such as substation reconnaissance, IED enumeration, control gateway compromise, telemetry exfiltration, and operator privilege escalation.

- [ ] **Step 2: Validate JSON syntax**

Run: `python -m json.tool aitads_augmented/configs/power-grid-minimal.json`

Expected: the command prints formatted JSON and exits with status 0.

### Task 2: Document the Scenario Mapping

**Files:**
- Create: `docs/power-grid-minimal-dataset.md`

- [ ] **Step 1: Write dataset notes**

Document that this is a minimal configuration-level reconstruction, not a field-level rewrite. Explain that no raw alert fragments are copied and that the model still observes the original AIT-ADS token values.

- [ ] **Step 2: Include attack-label mapping**

List how the existing AIT-ADS labels are interpreted for the power-grid setting:

```text
service_scan/dns_scan -> substation or control-network reconnaissance
wpscan/dirb -> IED/gateway asset and service enumeration
webshell_cmd -> control gateway compromise
crack_passwords -> operator account brute force
attacker_change_user -> operator role switch
escalated_sudo_command -> control privilege escalation
dnsteal_* -> telemetry or PMU data exfiltration
```

### Task 3: Verify Dataset Loading

**Files:**
- Read: `alertbert/aitads.py`
- Read: `aitads_augmented/configs/power-grid-minimal.json`

- [ ] **Step 1: Run split loading smoke test**

Run:

```bash
conda run -n alertbert python -c "from alertbert.aitads import AITAlertDataset; [print(split, len(AITAlertDataset(split=split, configuration='power-grid-minimal'))) for split in ['train','val','test']]"
```

Expected: all three splits load and print positive lengths.

- [ ] **Step 2: Run an all-split smoke test**

Run:

```bash
conda run -n alertbert python -c "from alertbert.aitads import AITAlertDataset; d=AITAlertDataset(split='all', configuration='power-grid-minimal'); print(len(d), d.keys, d.n_scenarios)"
```

Expected: the dataset loads, prints the feature keys, and reports four scenarios.

