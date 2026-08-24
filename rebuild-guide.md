# Full rebuild guide — Ubuntu on the PCIe5 NVMe (in order)

**Your setup / decisions:**
MSI MAG X670E Tomahawk WiFi (board code **7E12**) · Ryzen 7900X3D · displays on the
**AMD iGPU** (motherboard ports) · **RTX 3090 Ti = compute-only** (CUDA, no display) ·
new PCIe5 NVMe as boot drive, **unencrypted** · old PCIe4 drive **kept intact** as a
fallback until the new install is proven.

Do these phases top to bottom. Don't touch firmware or wipe anything until Phases
1–3 (all done while your current, known-working system is still running) are complete.

---

## Phase 0 — Choose your Ubuntu version
- [ ] **26.04.1 LTS** — recommended, since your display is AMD (the NVIDIA/Wayland
      concern doesn't apply to you). Longest support, best amdgpu + X3D scheduling.
      Just confirm the CUDA toolkit version you need lists 26.04 / kernel 7.0 support.
- [ ] **24.04.x LTS** — conservative alternative: keeps an X11 session, widest
      CUDA/ML-doc compatibility today. Pick this if you want zero tooling friction now.

The rest of this guide works for either — "the ISO" means whichever you chose.

---

## Phase 1 — Diagnose (current system still running)
Last cheap chance to catch a hardware problem a reinstall won't fix.
- [ ] `sudo bash diagnose.sh | tee diagnose-report.txt`
- [ ] Check **SMART "Percentage Used"** on BOTH NVMe drives — well under 100.
- [ ] Check **OpenGL renderer** under GRAPHICS — if it says NVIDIA, the old box was
      wrongly rendering the desktop on the dGPU (confirms the misconfig you suspected).
- [ ] Skim **previous-boot errors** (shown after a freeze) for anything alarming.

## Phase 2 — Audit startup scripts, cron, timers (#7)
- [ ] `bash audit-startup.sh | tee startup-audit-user.txt`
- [ ] `sudo bash audit-startup.sh | tee startup-audit-root.txt`
- [ ] Keep both files — this is your list of what to *deliberately* re-create later.
      Don't plan to copy it forward wholesale; that's the cruft you're escaping.

## Phase 3 — Back up everything (and record your drives)
**Record drive identity first** so you can't wipe the wrong disk later:
- [ ] `lsblk -o NAME,SIZE,TYPE,MODEL,SERIAL,MOUNTPOINT` → write down the PCIe5 drive's
      **MODEL + SERIAL + SIZE**.

**Back up the PCIe5 drive's current contents** (the install will erase it):
- [ ] `rsync -aHAX --info=progress2 /mount/of/pcie5/ /somewhere/safe/`

**Back up what you'll carry over from the old system:**
- [ ] Dotfiles / config: `~/.config`, `~/.local`, shell rc files, `~/.vscode`, Claude Code config
- [ ] Keys (do deliberately):
      `cp -a ~/.ssh ~/backup/ssh`
      `gpg --export-secret-keys --armor > ~/backup/gpg-secret.asc`
      `gpg --export --armor > ~/backup/gpg-public.asc`
- [ ] Browser profiles: `~/.mozilla`, `~/.config/google-chrome`, etc.
- [ ] Package inventory (to rebuild env, not restore blindly):
      `apt-mark showmanual > ~/backup/apt-manual.txt`
      `snap list > ~/backup/snap-list.txt`
      `pip freeze > ~/backup/pip-freeze.txt` (per venv) · `conda env export -n <env> > ~/backup/<env>.yml`
- [ ] Project repos (confirm pushed to git), local datasets, model checkpoints, databases
- [ ] Customized `/etc` bits you'll want to reference: `fstab`, `hosts`, docker config, udev rules
- [ ] The two `startup-audit-*.txt` files from Phase 2
- [ ] **Verify a backup actually restores** before proceeding.

## Phase 4 — Flash the BIOS (before the OS install)
- [ ] Snap a **photo of your current BIOS settings** — flashing resets to defaults.
- [ ] On MSI's official support page for **MAG X670E Tomahawk WiFi (7E12)**, download the
      latest **release** (non-beta) BIOS. Prefer stable unless a beta changelog fixes a
      specific problem you have. Confirm the file is for **7E12**.
- [ ] Put the file on a **FAT32** USB stick (a small one — not your future install stick).
- [ ] Reboot → enter BIOS → **M-FLASH** → select the file → let it flash and reboot.
- [ ] **First boot after flashing may sit 30+ sec** doing DDR5 memory retraining — this is
      normal, do NOT force-restart through it.

## Phase 5 — Configure the BIOS
- [ ] Set **Primary / Initial Display Output = Integrated Graphics (IGD/iGPU)**.
- [ ] Ensure the **iGPU is Enabled/Forced**, not "Auto" (Auto often disables it when a
      dGPU is present — a likely cause of your old display trouble).
- [ ] Enable **EXPO** memory profile (if you run it).
- [ ] Confirm boot mode is **UEFI**. Set fTPM as you prefer.
- [ ] Save & exit.

## Phase 6 — Make the install USB
- [ ] Download the Desktop ISO (amd64) for your chosen version from the official Ubuntu
      releases page, plus `SHA256SUMS` (+ `SHA256SUMS.gpg`) from the same directory.
- [ ] Verify: `sha256sum -c SHA256SUMS 2>&1 | grep OK`
- [ ] Write to an **8 GB+** stick with the built-in **Startup Disk Creator**
      (`usb-creator-gtk`), or with dd (triple-check the target is the USB, not an nvme):
      `sudo dd if=ubuntu-XX.XX.iso of=/dev/sdX bs=4M status=progress oflag=direct && sync`

## Phase 7 — Install Ubuntu
- [ ] Boot the USB (F-key boot menu) → "Try or Install".
- [ ] At disk selection, pick the **PCIe5 drive by the MODEL/SERIAL you recorded in Phase 3**.
- [ ] "Erase disk and install" **on that drive only**. **Leave encryption UNCHECKED.**
- [ ] Bootloader → install to the **new drive's EFI partition** (keeps the old PCIe4 drive
      independently bootable as your fallback).
- [ ] Finish install → in BIOS set the **new drive first** in boot order.

## Phase 8 — Post-install: make the iGPU/compute split real
- [ ] Confirm the desktop renders on AMD:
      `glxinfo | grep -i "OpenGL renderer"` → want AMD/Radeon, **not** NVIDIA
      (install `mesa-utils` first if glxinfo is missing).
- [ ] Install the NVIDIA driver for **compute only** — do NOT add an xorg.conf forcing it:
      `ubuntu-drivers devices` → `sudo ubuntu-drivers install`
- [ ] Verify: `nvidia-smi` shows the 3090 Ti (available for CUDA); display stays on the iGPU.
- [ ] If you use Docker with the GPU: install `nvidia-container-toolkit`.

## Phase 9 — Restore selectively & rebuild
- [ ] Restore dotfiles, keys, browser profiles from Phase 3.
- [ ] Rebuild your dev env from the inventory (conda/uv/pyenv, Docker, VS Code, Claude Code, CUDA).
- [ ] Re-create **only** the cron jobs / startup units you actually want, from the audit files.
- [ ] Audio check: `pactl info` should show **PipeWire**; test Bluetooth headphone auto-switch.
- [ ] Confirm the old annoyances are gone: second monitor works on boot, no boot freeze,
      audio routes correctly.

## Phase 10 — Validate, then reclaim the old drive
- [ ] Run the new install for ~a week. If anything's off, re-run `diagnose.sh`.
- [ ] Once proven, wipe/repurpose the old PCIe4 drive (fast scratch space for
      datasets/checkpoints is a great use).

---

### Quick gotcha list
- Pick the install target by **serial**, not size or name — two NVMe drives are easy to mix up.
- **Uncheck encryption** in the installer.
- The long post-BIOS-flash boot is **memory retraining**, not a hang.
- Ignore "update the chipset driver" in MSI changelogs — that's Windows-only.
- Keep the **old drive intact** until the new setup is fully validated.
