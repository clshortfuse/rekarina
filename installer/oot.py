"""Rekarina setup: OoT (Switch 2) from the player's own dump to a folder to play from.

    python -m installer.oot --game <file.nup or its folder> --dest <install folder> [--lang ru] [--prompts auto]
                            [--toolchain DIR] [--temp DIR] [--root DIR] [--dlls DIR]

The player names the folder with their dump; the setup finds the NUP container in it, checks that it is the version
the code patches were written for (the SHA-256 of every ExeFS file, tools/data/expected-oot.json), and then does what
the tools do by hand, as steps under the same rule as the TotK setup (runner.py: a step whose inputs
did not change since it was recorded is skipped, so a second run continues or updates):

    oot_check     the dump, the tools, the disk space
    oot_extract   the ExeFS out of the container (tools/oot_exefs.py): as stored for the lifter, uncompressed for the game
    oot_lift      the game code to C (recomp/gen.py with recomp/patches-oot-v111.txt)
    oot_compile   the C and the runtime to totk.exe (tools/build_game.py --version oot)
    oot_components  the upscaler libraries this machine's graphics card can use (DLSS, frame generation, FSR 3.1),
                  downloaded from NVIDIA's and AMD's own repositories (components.py); a failure only warns
    oot_prompts   the button prompt overlays and fonts (a minute)
    oot_prepare   the shaders translated before play: every variant, in the foreground (the driver's build of them is
                  the game's start and the launcher's background work, background_main)
    oot_package   the folder to play from (tools/package_oot.py): Rekarina.exe, its libraries, caches and rekarina.ini,
                  a new rekarina.ini with first settings for the graphics card (components.preset), and the list of
                  what the launcher does after the setup (plan_background)

`--background` runs that list instead (the launcher's background job, background_main).

The RomFS stays in the player's NUP: the game reads it in place, so the dump must stay where it is. Prints JSON events
on stdout like installer/cli.py (events.py), and after "start" one "eta" event with the time each step should take
(eta.py); the window (gui_oot.py) reads them. Exit code: 0 done, 1 failed, 2 bad command line, 3 cancelled.

Where the tools come from: a release folder brings its own llvm, ninja and libraries (layout.py). In a checkout the
compiler is the one --toolchain names (a folder with llvm/bin/clang and bin/ninja, such as an unpacked TotK-Setup
release), else REKARINA_TOOLCHAIN, else toolchain/llvm-mingw-* of the checkout, else the newest TotK-Setup release
found beside the checkout's parent folder.
"""
import argparse
import ast
import hashlib
import io
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from . import components, eta, fork, mods, net
from . import state as statefile
from .events import Reporter
from .runner import Context, InstallError, Step, run_steps

EXIT = {'ok': 0, 'failed': 1, 'cancelled': 3}
WIN = os.name == 'nt'
EXE = '.exe' if WIN else ''
GAME_EXE = 'Rekarina' + EXE
INI = 'rekarina.ini'      # the settings file beside the game
OLD_INI = 'totk.ini'      # its name in a folder made before 1.0.0: the game and package_oot.py rename it
SHIM = 'rekarina_dlssg.dll'   # our library between the game and Streamline, as the game folder and a release name it
RENHDRUM = 'renhdrum.dll'     # RenHDrum's controller haptics library (tools/build_renhdrum.py, runtime/rumble_rh.c)
STYLES = (('switch', 'switch'), ('xbox', 'xbox'), ('xbox-label', 'xbox-label'))   # shared-fonts-<folder>, --style
# Libraries the game loads when they lie beside it (DLSS, FSR 3.1, frame generation). They are not ours to ship: the
# components step downloads them from their vendors; --dlls names a folder with the player's own copies, which then
# take the place of the downloaded ones.
OPTIONAL_DLLS = ('nvngx_dlss.dll', 'amd_fidelityfx_loader_vk.dll', 'amd_fidelityfx_upscaler_vk.dll',
                 'amd_fidelityfx_framegeneration_vk.dll', SHIM, 'totk_dlssg.dll', 'sl.interposer.dll', 'sl.common.dll',
                 'sl.dlss.dll', 'sl.dlss_g.dll', 'sl.reflex.dll', 'sl.pcl.dll', 'nvngx_dlssg.dll', 'NvLowLatencyVk.dll')
NEED_DEST = 14 << 30   # the exe, the shader and driver caches, the overlays
NEED_TEMP = 9 << 30    # the generated C, the objects, the build's copy of the caches


# ---- the dump the game reads, and the game exe itself
# The game reads its data from the player's .nup at every start, from the path in rekarina.ini ([paths] romfs, written
# by tools/package_oot.py with a new file). Two players of 1.2.5 moved or renamed the dump after the setup, or had it in
# a Documents folder OneDrive keeps online only, and the game ended at its start (runtime/romfs.c dump_missing). The
# launcher notices it (dump_problem) and asks where the file is now; the setup's check then writes the new path into
# the file (set_ini_dump), without building anything again.
ROMFS_LINE = re.compile(r'^(\s*)((?:rekarina_|totk_)?romfs)(\s*=\s*)(.*?)\s*$', re.IGNORECASE)
FILE_ATTRIBUTE_OFFLINE = 0x1000
FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x40000
FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x400000   # a cloud file (OneDrive "online only"): its data is not on this PC


def game_ini(dest):
    """The game folder's settings file: rekarina.ini, or totk.ini in a folder made before 1.0.0; None when neither."""
    for name in (INI, OLD_INI):
        if (Path(dest) / name).is_file():
            return Path(dest) / name
    return None


def ini_dump(dest):
    """The .nup the game in dest reads (its settings file's romfs line, the last one as the game takes it); None when
    the file or the line is not there."""
    path = game_ini(dest)
    if path is None:
        return None
    try:
        text = path.read_text(encoding='utf-8-sig', errors='replace')
    except OSError:
        return None
    found = None
    for line in text.splitlines():
        m = ROMFS_LINE.match(line)
        if m and m.group(4):
            found = m.group(4)
    return Path(found) if found else None


def set_ini_dump(dest, nup):
    """The settings file of dest names nup on its romfs line (the line written in place, every other byte kept; a file
    without the line gets one at the top of its [paths] section, or a new section). True when the file changed; False
    when it already named nup or there is no settings file (a new one is written with the path, package_oot.py)."""
    path = game_ini(dest)
    if path is None:
        return False
    data = path.read_bytes()
    bom = b'\xef\xbb\xbf' if data.startswith(b'\xef\xbb\xbf') else b''
    text = data[len(bom):].decode('utf-8', errors='surrogateescape')
    eol = '\r\n' if '\r\n' in text else '\n'
    lines = text.split(eol)
    value = Path(nup).as_posix()
    hits = [i for i, line in enumerate(lines) if ROMFS_LINE.match(line)]
    if hits:
        for i in hits:   # every romfs line: the game takes the last, and none may name the old place
            m = ROMFS_LINE.match(lines[i])
            lines[i] = f'{m.group(1)}{m.group(2)}{m.group(3)}{value}'
    else:
        at = next((i + 1 for i, line in enumerate(lines) if line.strip().lower() == '[paths]'), None)
        if at is None:
            lines[:0] = ['[paths]', f'romfs = {value}', '']
        else:
            lines.insert(at, f'romfs = {value}')
    new = bom + eol.join(lines).encode('utf-8', errors='surrogateescape')
    if new == data:
        return False
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(new)
    os.replace(tmp, path)
    return True


def dump_problem(dest):
    """Why the game in dest cannot read its dump, as (kind, path): 'missing' when no file is at the path its settings
    file names (moved, renamed, a disk not there), 'online' when the file is a cloud file whose data is not on this PC
    (OneDrive's "online only": the game's start would wait for the download, or fail when OneDrive is not running).
    None when the file is there, or the folder names none."""
    nup = ini_dump(dest)
    if nup is None:
        return None
    try:
        st = os.stat(nup)
    except (OSError, ValueError):
        return ('missing', nup)
    if not os.path.isfile(nup):
        return ('missing', nup)
    attrs = getattr(st, 'st_file_attributes', 0)
    if attrs & (FILE_ATTRIBUTE_OFFLINE | FILE_ATTRIBUTE_RECALL_ON_OPEN | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS):
        return ('online', nup)
    return None


SINGLE_TRAILER = b'RKSFX001'   # the end of the one-file Rekarina.exe (installer/launcher/rekarina_single.c read_trailer)
SINGLE_TRAILER_SIZE = 56


def is_single_launcher(path):
    """True when path is the downloaded one-file Rekarina.exe (the setup and its launcher in one file), which has the
    game's name: saved into the game folder, it takes the game's place, and Play then opens the launcher again."""
    try:
        with open(path, 'rb') as f:
            size = f.seek(0, os.SEEK_END)
            if size <= SINGLE_TRAILER_SIZE:
                return False
            f.seek(size - SINGLE_TRAILER_SIZE)
            tail = f.read(SINGLE_TRAILER_SIZE)
    except OSError:
        return False
    if not tail.startswith(SINGLE_TRAILER):
        return False
    offset, length = int.from_bytes(tail[8:16], 'little'), int.from_bytes(tail[16:24], 'little')
    return offset + length + SINGLE_TRAILER_SIZE == size

# The helpers below repeat installer/steps.py's (file_sha256, fingerprint, kill_tree, run_tool, free space): that module
# imports the Switch 1 extraction tools and their libraries, which this setup does not need.
NINJA_LINE = re.compile(r'^\[(\d+)/(\d+)\] ')
TRANSLATED = re.compile(r'gfx_shader: prepare: (\d+) of (\d+) pairs')   # totk.exe --prepare-shaders
LIBRARIES = re.compile(r'prebuild preparation (?:progress|done): (\d+) of (\d+) variants')   # its driver cache phase
RUN_SCRIPT = ('import os, runpy, sys; p = sys.argv[1]; sys.path.insert(0, os.path.dirname(p)); sys.argv = sys.argv[1:]; '
              "runpy.run_path(p, run_name='__main__')")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb', buffering=0) as f:
        buf = bytearray(8 << 20)
        view = memoryview(buf)
        while n := f.readinto(buf):
            h.update(view[:n])
    return h.hexdigest()


def fingerprint(*parts):
    """SHA-256 over strings and the contents of files (a missing file counts as missing). A (name, file) pair counts
    the file under that name: a file that was renamed keeps the fingerprints made under its old name."""
    h = hashlib.sha256()
    for p in parts:
        name = None
        if isinstance(p, tuple):
            name, p = p
        if isinstance(p, Path):
            h.update((name or p.name).encode() + b'\0' + (bytes.fromhex(file_sha256(p)) if p.is_file() else b'missing'))
        else:
            h.update(str(p).encode())
        h.update(b'\1')
    return h.hexdigest()


WORK_IN_DEST = '.setup-work'   # in the game folder: the working files of a setup whose own disk has no room for them


def work_folder(default, asked, dest):
    """(the folder of this game folder's working files, the free GiB of the setup's disk when that sent them into the
    game folder, else None). The files (the generated C, the objects, the build's caches: NEED_TEMP) live with the
    setup, in its temp folder, one sub-folder per game folder; --temp names another place. A setup that was unpacked on
    a disk without room for them (the single-file Rekarina.exe unpacks into the user's application data, on the system
    disk) puts them into the game folder instead, when that is on another disk with room for both: the game folder is
    the one place the player chose. Later runs go where the objects of the earlier one are, wherever the free space
    is by then, so an update compiles only what changed."""
    tag = f'oot-{hashlib.sha256(str(dest).lower().encode()).hexdigest()[:8]}'
    if asked:
        return Path(asked).resolve() / tag, None
    base = Path(default).resolve() if default else dest / WORK_IN_DEST
    here, there = base / tag, dest / WORK_IN_DEST / tag
    if here == there or (here / 'build' / 'obj').is_dir():
        return here, None
    if (there / 'build' / 'obj').is_dir():
        return there, None
    try:
        drive = lambda p: os.path.splitdrive(str(p))[0].lower()
        free, room = free_bytes(here), free_bytes(dest)
        if drive(here) != drive(dest.resolve()) and free < NEED_TEMP + (1 << 30) and room >= NEED_DEST + NEED_TEMP:
            return there, round(free / 2**30, 1)
    except OSError:
        pass
    return here, None


def free_bytes(path):
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def kill_tree(proc):
    if proc.poll() is not None:
        return
    if WIN:
        subprocess.run(['taskkill', '/T', '/F', '/PID', str(proc.pid)], capture_output=True)
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        proc.kill()


def extra_python_path(ctx):
    """PYTHONPATH for the tools when ensure_zbic had to unpack the package into the temp folder (a Python whose own
    site-packages is read-only, as inside an AppImage); nothing otherwise."""
    site = ctx.temp / 'site-packages'
    if not site.is_dir():
        return {}
    return {'PYTHONPATH': os.pathsep.join(p for p in (str(site), os.environ.get('PYTHONPATH', '')) if p)}


def run_tool(ctx, step, script, args, parse=None, env=None):
    """Run a tool script with the setup's Python. Its output goes to <temp>/logs/<step>-<script>.log and, line by line,
    to parse(line); env: variables added to the setup's own. Returns (exit code, log path). The tool runs with
    faulthandler, so a crash of the process (an access violation) leaves the Python stack where it happened in the
    log, not only an exit code."""
    logs = ctx.temp / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f'{step}-{Path(script).stem}.log'
    command = [str(ctx.layout.python), '-X', 'faulthandler', '-c', RUN_SCRIPT, str(script), *map(str, args)]
    ctx.reporter.log('run ' + ' '.join(command[5:]))
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', **extra_python_path(ctx), **(env or {}))
    with open(log, 'w', encoding='utf-8', newline='\n') as out:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ctx.layout.root, env=env,
                                text=True, encoding='utf-8', errors='replace', bufsize=1, start_new_session=not WIN)
        try:
            for line in proc.stdout:
                out.write(line)
                if parse:
                    parse(line.rstrip())
            code = proc.wait()
        except BaseException:
            kill_tree(proc)
            raise
    ctx.reporter.log(f'{Path(script).name}: exit code {code}; output in {log}')
    return code, log


def crashed(code):
    """Whether an exit code is a crash of the process, not an error the tool reported: a Windows exception status
    (0xC0000005 an access violation, 0xC0000374 a damaged heap, ...) or a POSIX signal."""
    return code < 0 or code >= 0xC0000000


def last_line(log):
    """The last line a tool wrote: the exception that ended it, or the place faulthandler found it crashed."""
    try:
        lines = [line.strip() for line in Path(log).read_text(encoding='utf-8', errors='replace').splitlines() if line.strip()]
    except OSError:
        return ''
    return lines[-1] if lines else ''


class Unstable(Exception):
    """A tool failed every time it ran and each time differently (tool_steady)."""

    def __init__(self, tool, code, log, tries):
        super().__init__(tool, code, log, tries)
        self.tool, self.code, self.log, self.tries = tool, code, log, tries

    def error(self):
        return InstallError('oot.unstable', tool=self.tool, tries=self.tries, code=self.code, log=str(self.log))


def tool_steady(ctx, step, script, args, parse=None, env=None, tries=3, within=None):
    """run_tool, again after a failure that need not come back. Returns (exit code, log path) as run_tool does: 0 once
    a run works, or the code of a failure that came twice the same way (the same code and last line: an error in the
    input or the tool, which a third run would repeat). Raises Unstable when every one of `tries` runs failed and they
    did not all fail alike. within: seconds; a run that failed after more than that is not run again (a long step).
    Why: a player's setup crashed in the button prompts' drawing, a pure Python tool that works on other machines with
    the same dump, a different way at each run (an access violation, Python objects of the wrong type in fontTools, an
    earlier release's shader tool the same way): memory damaged in that process, which another run may not meet. Each
    failed run's log is kept beside the last one (<log>.1.log, ...)."""
    seen = []
    for attempt in range(1, tries + 1):
        began = time.monotonic()
        code, log = run_tool(ctx, step, script, args, parse, env)
        if not code:
            return code, log
        sign = (code, last_line(log))
        if sign in seen or within is not None and time.monotonic() - began > within:
            return code, log
        seen.append(sign)
        if attempt == tries:
            raise Unstable(Path(script).name, code, log, tries)
        kept = log.with_name(f'{log.stem}.{attempt}.log')
        try:
            log.replace(kept)
        except OSError:
            kept = log
        ctx.reporter.log(f'{Path(script).name} failed ({"crashed, " if crashed(code) else ""}exit code {code}: '
                         f'{sign[1][:200]}); its log is {kept}; running it again ({attempt + 1} of {tries})')
    return code, log


# The last line of a tool that stopped on data of the dump it could not unpack (the zstd library's own words, the NUP
# reader's), and tools/nup_damage.py's report of one damaged file.
DUMP_DAMAGE = re.compile(r'ZstdError|Data corruption|corrupted|NupError|size mismatch for|short read for', re.I)
DAMAGED_LINE = re.compile(r'^damaged: (.+)$')


def dump_damage(ctx, step, log, scope):
    """After a tool that reads the dump failed on data it could not unpack (DUMP_DAMAGE in its last line): which files
    of the .nup are damaged (tools/nup_damage.py, scope '--shaders' or '--all'), named with where they lie in it, as
    InstallError check.input_damaged: the .nup has to be copied or dumped again. A step cannot leave such a file out,
    since the game reads the same files while it runs. Returns when the check finds nothing (the caller then raises
    its own error). Why: a player's shader step ended in "ZstdError: ... Data corruption detected" and a traceback that
    named no file (1.2.8)."""
    tool = ctx.layout.tools / 'nup_damage.py'
    if not DUMP_DAMAGE.search(last_line(log)) or not tool.is_file():
        return
    ctx.reporter.progress(step, 'diagnose', force=True)
    found = []

    def parse(line):
        m = DAMAGED_LINE.match(line)
        if m:
            found.append(m.group(1))

    code, check_log = run_tool(ctx, step, tool, [ctx.nup, scope], parse)
    if code == 3 and found:
        more = f'; {len(found) - 3} more in {check_log}' if len(found) > 3 else ''
        raise InstallError('check.input_damaged', input='.nup', path=str(ctx.nup), detail='; '.join(found[:3]) + more)
    ctx.reporter.log(f'nup_damage.py found no damaged file (exit code {code}); its log is {check_log}')



def cpu_level():
    """The level the game is compiled for on this machine (tools/build_game.py --cpu): 'v3', the default, needs AVX2
    and FMA; a processor without AVX2 gets 'v2' (SSE4.2), slower, where the default build would stop at its first
    AVX2 instruction. REKARINA_CPU names one for a test."""
    forced = os.environ.get('REKARINA_CPU')
    if forced in ('v2', 'v3'):
        return forced
    try:
        if os.name == 'nt':
            import ctypes
            return 'v3' if ctypes.windll.kernel32.IsProcessorFeaturePresent(40) else 'v2'   # PF_AVX2_INSTRUCTIONS_AVAILABLE
        with open('/proc/cpuinfo', encoding='ascii', errors='replace') as f:
            return 'v3' if ' avx2' in f.read() else 'v2'
    except (OSError, AttributeError):
        return 'v3'


@dataclass(frozen=True)
class OotLayout:
    root: Path
    dev: bool
    release: str
    python: Path
    tools: Path
    recomp: Path
    llvm: Path           # the tree with bin/clang
    ninja: Path
    deps: Path
    xlate: Path
    temp: Path
    pgo: Path | None     # the execution profile the runtime is compiled with, when there is one
    # A Linux release: the Steam Runtime sysroot the game compiles and links against (llvm/sysroot, as the TotK setup's
    # installer/layout.py), and the libraries the game finds beside itself (bin/libSDL3.so.0); a checkout uses the host's
    sysroot: Path | None = None
    libs: tuple = ()

    @property
    def runtime(self):   # the icons the window shows
        return self.root / 'runtime'

    @property
    def shim(self):
        """Our own library between the game and Streamline (runtime/dlssg): a release carries it in bin/ as
        rekarina_dlssg.dll, a checkout has it where its build put it, as totk_dlssg.dll."""
        return self.root / 'build' / 'oot' / 'totk_dlssg.dll' if self.dev else self.root / 'bin' / SHIM

    @property
    def renhdrum(self):
        """RenHDrum's library (tools/build_renhdrum.py), which the game's Vibration setting plays controllers through:
        a release carries it in bin/, a checkout has it where that build put it."""
        return self.root / 'build' / 'oot' / RENHDRUM if self.dev else self.root / 'bin' / RENHDRUM

    @property
    def patches(self):
        return self.recomp / 'patches-oot-v111.txt'

    @property
    def expected(self):
        return self.tools / 'data' / 'expected-oot.json'

    def missing(self):
        need = [self.python, self.tools / 'oot_exefs.py', self.tools / 'build_game.py', self.tools / 'oot_prepare_shaders.py',
                self.tools / 'make_oot_prompts.py', self.tools / 'make_ext_font.py', self.tools / 'package_oot.py',
                self.recomp / 'gen.py', self.patches, self.expected, self.llvm / 'bin' / ('clang' + EXE), self.ninja,
                self.deps / 'include', self.deps / 'lib', self.xlate, *([self.sysroot] if self.sysroot else []), *self.libs]
        return [str(p) for p in need if not p.exists()]


def toolchain_root(root, named):
    """The folder that holds llvm/ and bin/ninja (see the module text)."""
    for cand in (named, os.environ.get('REKARINA_TOOLCHAIN')):
        if cand:
            return Path(cand)
    trees = sorted(p.parent.parent for p in (root / 'toolchain').glob('llvm-mingw-*/llvm-mingw-*/bin/clang' + EXE))
    if trees:
        return trees[-1].parent
    setups = sorted(root.parent.parent.glob('TotK-Setup-*/release/TotK-Setup-*'))
    return setups[-1] if setups else root / 'toolchain' / 'missing'


def find_layout(root=None, toolchain=None, temp=None):
    root = Path(root or Path(__file__).resolve().parent.parent).resolve()
    version = (root / 'VERSION').read_text(encoding='utf-8').strip() if (root / 'VERSION').is_file() else 'unknown'
    bundled = root / 'python' / ('python.exe' if WIN else 'bin/python3')
    if bundled.is_file():   # a release folder
        return OotLayout(root=root, dev=False, release=version, python=bundled, tools=root / 'tools', recomp=root / 'recomp',
                         llvm=root / 'llvm', ninja=root / 'bin' / ('ninja' + EXE), deps=root / 'runtime' / 'deps',
                         xlate=root / 'bin' / ('rekarina_xlate' + EXE), temp=Path(temp) if temp else root / 'temp',
                         pgo=root / 'data' / 'pgo-oot.profdata',
                         sysroot=None if WIN else root / 'llvm' / 'sysroot',
                         libs=() if WIN else (root / 'bin' / 'libSDL3.so.0',))
    tc = toolchain_root(root, toolchain)
    llvm = tc / 'llvm' if (tc / 'llvm').is_dir() else tc
    ninja = tc / 'bin' / ('ninja' + EXE)
    return OotLayout(root=root, dev=True, release=f'{version}-dev', python=Path(sys.executable), tools=root / 'tools',
                     recomp=root / 'recomp', llvm=llvm, ninja=ninja if ninja.is_file() else llvm / 'bin' / ('ninja' + EXE),
                     deps=root / 'toolchain' / 'deps-mingw', xlate=root / 'build' / 'xlate' / ('totk_xlate' + EXE),
                     temp=Path(temp) if temp else root / 'build' / 'install-temp', pgo=root / 'build' / 'pgo' / 'pgo-oot.profdata')


# The game's main module is stored as ZBIC (the platform's zstd variant, system 22.0.0 on), which the tools read with the
# zbic package (recomp/nso.py). It is GPL software, so no release folder carries it: a Python that lacks it gets the
# wheel for its version from the Python Package Index, checked against the index's own SHA-256, as a player's own
# download (like the upscaler libraries). A source checkout installs it with its other requirements.
ZBIC_INDEX = 'https://pypi.org/pypi/zbic/json'


def ensure_zbic(ctx, step):
    python = str(ctx.layout.python)
    quiet = dict(capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW if WIN else 0)
    if subprocess.run([python, '-c', 'import zbic'], env=dict(os.environ, **extra_python_path(ctx)), **quiet).returncode == 0:
        return
    ctx.reporter.progress(step, 'zbic', force=True)
    probe = subprocess.run([python, '-c', 'import sys, sysconfig; print(f"cp{sys.version_info[0]}{sys.version_info[1]}"); '
                            'print(sysconfig.get_paths()["purelib"])'], **quiet)
    lines = probe.stdout.splitlines()   # one per line: the folder may hold spaces
    try:
        # the wheel of this platform: win_amd64, or on x86-64 Linux the manylinux one (glibc 2.17, older than the
        # setup's own floor of 2.34)
        plat = 'win_amd64' if WIN else 'manylinux' if sys.platform.startswith('linux') and platform.machine() == 'x86_64' else None
        if probe.returncode or len(lines) < 2 or not plat:
            raise OSError('no wheel is known for this Python; install the package with: pip install zbic')
        tag, site = lines[0], Path(lines[1])
        # (through Windows' own TLS when Python's certificate check fails: net.fetch_bytes; the wheel is checked
        # against the index's checksum either way)
        data, how = net.fetch_bytes(ZBIC_INDEX, 30, ctx.reporter.log)
        meta = json.loads(data)
        wheels = [f for f in meta.get('urls', []) if f['filename'].endswith('.whl') and (
            f['filename'].endswith(f'-{tag}-{tag}-win_amd64.whl') if WIN else
            f'-{tag}-{tag}-manylinux' in f['filename'] and f['filename'].endswith('_x86_64.whl'))]
        if not wheels:
            raise OSError(f'the index has no {tag} {plat} wheel of zbic {meta.get("info", {}).get("version", "")}')
        data, how = net.fetch_bytes(wheels[0]['url'], 60, ctx.reporter.log)
        if hashlib.sha256(data).hexdigest() != wheels[0]['digests']['sha256']:
            raise OSError('the downloaded wheel does not match the checksum the index gives')
        ctx.reporter.log(f'zbic: {wheels[0]["filename"]} downloaded ({how}), its checksum is the index\'s')
        parent = site
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if not os.access(parent, os.W_OK):   # the Python lives on a read-only mount (an AppImage): into the temp folder
            site = ctx.temp / 'site-packages'
        site.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                if not name.endswith('/') and '..' not in Path(name).parts and not Path(name).is_absolute():
                    z.extract(name, site)
        if subprocess.run([python, '-c', 'import zbic'], env=dict(os.environ, **extra_python_path(ctx)), **quiet).returncode:
            raise OSError('the package was unpacked but does not load')
    except (ValueError, KeyError, zipfile.BadZipFile, *net.NET_ERRORS) as e:
        raise InstallError('oot.zbic_failed', reason=str(e)) from None


def tool_json(ctx, step, script, args):
    """Run a tool that prints one JSON object as its last line; (object or None, log path)."""
    last = ['']

    def parse(line):
        if line.startswith('{'):
            last[0] = line

    code, log = run_tool(ctx, step, script, args, parse)
    try:
        return json.loads(last[0]), log
    except ValueError:
        return ({'error': 'tool', 'code': code} if code else None), log


def tree_stamp(*paths):
    """A fingerprint of source trees: the path and the contents of every file. Contents, not modification times: a
    release unpacked on another day, or by the launcher's update, has new times on files that did not change, and
    the steps that read them must then be skipped, not run again (a compile and a shader preparation for nothing).
    The trees are small (a release's runtime folder is 50 MiB: a fraction of a second)."""
    h = hashlib.sha256()
    for p in paths:
        files = sorted(f for f in p.rglob('*') if f.is_file() and '__pycache__' not in f.parts) if p.is_dir() else [p]
        for f in files:
            name = f.relative_to(p).as_posix() if p.is_dir() else f.name
            h.update(name.encode() + bytes(1) + (bytes.fromhex(file_sha256(f)) if f.is_file() else b'missing') + bytes([1]))
    return h.hexdigest()


def code_text(path):
    """What a source file says as code, for a step's key: a Python file's syntax tree without its docstrings (comments
    are not in it), a patch file's lines as recomp/patches.py reads them (cut at '#', words joined by one space, blank
    lines left out), any other file its bytes. A comment or a docstring edited in the lifter or a drawing tool then
    makes no player translate the game code again (the lift: minutes, and the compile of every chunk that changed)."""
    data = Path(path).read_bytes()
    suffix = Path(path).suffix
    if suffix == '.py':
        try:
            tree = ast.parse(data)
        except (SyntaxError, ValueError):
            return data
        for node in ast.walk(tree):
            body = getattr(node, 'body', None)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body and \
                    isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
        return b'py-ast\0' + ast.dump(tree, include_attributes=False).encode()
    if suffix == '.txt':
        lines = [' '.join(line.split('#', 1)[0].split()) for line in data.decode('utf-8', errors='replace').splitlines()]
        return b'patch-lines\0' + '\n'.join(line for line in lines if line).encode()
    return data


def code_stamp(*paths):
    """tree_stamp by code_text: the path and what every file says as code."""
    h = hashlib.sha256()
    for p in paths:
        files = sorted(f for f in p.rglob('*') if f.is_file() and '__pycache__' not in f.parts) if p.is_dir() else [p]
        for f in files:
            name = f.relative_to(p).as_posix() if p.is_dir() else f.name
            h.update(name.encode() + bytes(1) + (hashlib.sha256(code_text(f)).digest() if f.is_file() else b'missing') +
                     bytes([1]))
    return h.hexdigest()


def host_gpu(ctx):
    """The graphics card of this machine (components.detect), asked once per run; None when nothing answered."""
    if not hasattr(ctx, 'gpu'):
        try:
            ctx.gpu = components.detect()
        except Exception as e:   # a detection that breaks must cost the presets, not the setup
            ctx.reporter.log(f'components: the graphics card could not be detected: {type(e).__name__}: {e}')
            ctx.gpu = None
    return ctx.gpu


class Check(Step):
    id = 'oot_check'
    always = True

    def run(self, ctx):
        layout, rep, opt = ctx.layout, ctx.reporter, ctx.options
        missing = layout.missing()
        if missing:
            raise InstallError('oot.tools_missing', paths=', '.join(missing))
        ensure_zbic(ctx, self.id)
        rep.progress(self.id, 'dump', force=True)
        game = opt.game or (ctx.state.get('oot') or {}).get('nup')
        if not game:
            raise InstallError('oot.need_game')
        info, _ = tool_json(ctx, self.id, layout.tools / 'oot_exefs.py', [game])
        if not info or info.get('error') in ('tool', 'bad_nup'):
            raise InstallError('oot.bad_nup', path=str(game), detail=(info or {}).get('detail', ''))
        if info.get('error') == 'no_nup':
            raise InstallError('oot.no_nup', path=str(game))
        if info.get('error'):
            raise InstallError('oot.no_exefs', path=info.get('path', str(game)))
        expected = json.loads(layout.expected.read_text(encoding='utf-8'))
        other = sorted(n for n, h in expected['exefs'].items() if info['files'].get(n) != h)
        if other:
            raise InstallError('oot.other_version', path=info['nup'], version=expected['version'], files=', '.join(other))
        ctx.nup, ctx.exefs_hashes = Path(info['nup']), info['files']
        ctx.state['oot'] = {'nup': info['nup'], 'version': expected['version']}
        # The game reads the dump from the path in its settings file: a dump chosen anew (moved, renamed, or the
        # launcher's question after the game could not find it) goes in there now. The steps' keys do not hold the
        # path, so nothing is built again for it.
        try:
            if set_ini_dump(ctx.dest, ctx.nup):
                rep.log(f'{game_ini(ctx.dest).name}: the game now reads its data from {ctx.nup}')
        except OSError as e:
            raise InstallError('cleanup.ini_failed', ini_file=INI, detail=str(e)) from None
        rep.progress(self.id, 'space', force=True)
        ctx.dest.mkdir(parents=True, exist_ok=True)
        ctx.temp.mkdir(parents=True, exist_ok=True)
        first = not (ctx.dest / GAME_EXE).is_file()
        need, temp_need = (NEED_DEST if first else 2 << 30), (NEED_TEMP if not (ctx.temp / 'build' / 'obj').is_dir() else 2 << 30)
        free, temp_free = free_bytes(ctx.dest), free_bytes(ctx.temp)
        same = os.path.splitdrive(str(ctx.dest.resolve()))[0].lower() == os.path.splitdrive(str(ctx.temp.resolve()))[0].lower()
        rep.emit('space', dest=str(ctx.dest), need=need + (temp_need if same else 0), free=free, temp=str(ctx.temp),
                 temp_need=0 if same else temp_need, temp_free=temp_free, same_disk=same)
        if free < need + (temp_need if same else 0) or (not same and temp_free < temp_need):
            raise InstallError('oot.no_space', path=str(ctx.dest if same or free < need else ctx.temp),
                               need=round((need + (temp_need if same else 0)) / 2**30, 1), free=round(free / 2**30, 1))
        return {}


class Extract(Step):
    id = 'oot_extract'

    def key(self, ctx):
        expected = json.loads(ctx.layout.expected.read_text(encoding='utf-8'))
        return fingerprint('oot-extract-1', json.dumps(expected['exefs'], sort_keys=True), ctx.layout.tools / 'nso_plain.py')

    def current(self, ctx, record):
        sizes = record.get('sizes') or {}
        return bool(sizes) and all((ctx.dest / 'exefs' / n).is_file() and (ctx.dest / 'exefs' / n).stat().st_size == s and
                                   (ctx.temp / 'exefs' / n).is_file() for n, s in sizes.items())

    def run(self, ctx):
        ctx.reporter.progress(self.id, 'exefs', force=True)
        info, log = tool_json(ctx, self.id, ctx.layout.tools / 'oot_exefs.py',
                              [ctx.nup, '--out', ctx.temp / 'exefs', '--plain', ctx.dest / 'exefs'])
        if not info or info.get('error'):
            raise InstallError('oot.extract_failed', log=str(log))
        return {'sizes': {n: (ctx.dest / 'exefs' / n).stat().st_size for n in info['files']}}


def mod_patch_files(ctx):
    """The files of the mods folder that change the game's code: each mod's cheat files and its ExeFS patch files
    (.ips, .pchtxt in its exefs folder and in atmosphere/exefs_patches/<name>/), as tools/cheats.py and
    tools/exefs_patches.py find them. All of them, whatever build they are for: the tools sort that out."""
    return [f for m in mods.scan(ctx.dest) for f in mods.patch_files(m)]


def status_file(ctx, name):
    """build-cache/<name>.status in the game folder: what the build holds of the mods' cheats (tools/cheats.py) or
    ExeFS patches (tools/exefs_patches.py). The game reads it (runtime/cheats.c, runtime/mods.c)."""
    return ctx.dest / 'build-cache' / f'{name}.status'


def lift_base_key(ctx):
    """The lift's key without the mods' files: what it was before mods were built in, and what a game folder without
    code patches still has, so nothing is built again for it."""
    # (by code since 1.2.8: an edited comment of the patch file or the lifter lifts nothing again; 'oot-lift-2' held the
    # bytes)
    return fingerprint('oot-lift-4', ctx.keys['oot_extract'], code_stamp(ctx.layout.patches), code_stamp(ctx.layout.recomp))


def compile_base_key(ctx):
    """The compile step's key for the lift without mods. The shaders are prepared for the game's code, not for the
    few sites a mod patches, so the shader step stands on this key and a mod costs no second preparation."""
    layout = ctx.layout
    return fingerprint('oot-compile-2', lift_base_key(ctx), tree_stamp(layout.root / 'runtime', layout.tools / 'build_game.py'),
                       tree_stamp(layout.pgo) if layout.pgo and layout.pgo.is_file() else 'no profile', cpu_level())


class Lift(Step):
    id = 'oot_lift'

    def key(self, ctx, base=None):
        files = mod_patch_files(ctx)
        base = base or lift_base_key(ctx)
        if not files:
            return base
        # a file's name and content, and its place under mods/: the tools name each patch set after its mod folder
        return fingerprint('oot-lift-3', base, ctx.layout.tools / 'cheats.py', ctx.layout.tools / 'exefs_patches.py',
                           *files, *(f.relative_to(mods.mods_dir(ctx.dest)).as_posix() for f in files))

    def adopts(self, ctx, record):
        """A record of the key's form before 1.2.8 ('oot-lift-2': the patch file and the lifter by their bytes) stands
        when those bytes are still the ones it was lifted from: the key's form changed, not the lift."""
        legacy = fingerprint('oot-lift-2', ctx.keys['oot_extract'], ctx.layout.patches, tree_stamp(ctx.layout.recomp))
        return record.get('key') == self.key(ctx, legacy)

    def current(self, ctx, record):
        return (ctx.temp / 'gen' / 'lift.done').is_file() and \
            (ctx.temp / 'gen' / 'lift.done').read_text(encoding='utf-8').strip() == record.get('key')

    def run(self, ctx):
        rep, gen = ctx.reporter, ctx.temp / 'gen'
        phase = ['discover']

        def parse(line):   # recomp/gen.py's own lines: functions found, then files written
            if 'emit' in line.lower() or 'writing' in line.lower():
                phase[0] = 'emit'
            rep.progress(self.id, phase[0])

        rep.progress(self.id, 'discover', force=True)
        (gen / 'lift.done').unlink(missing_ok=True)
        # The code cheats and the ExeFS patch files of the mods become patch sets beside the port's own; the status
        # files tell the game what the build holds. They are the player's downloads: files the tools refuse cost
        # those cheats or patches, not the build, which then goes on without them and says so.
        plain, patches, skipped, inputs = ctx.layout.patches, ctx.layout.patches, {}, mod_patch_files(ctx)
        stages = [('cheats.py', 'patches-cheats.txt', status_file(ctx, 'cheats'), 'warn.cheats_skipped'),
                  ('exefs_patches.py', 'patches-exefs.txt', status_file(ctx, 'exefs'), 'warn.exefs_skipped')]
        for tool, out, status, warn in stages:
            status.unlink(missing_ok=True)   # no status: the game takes every such file as not built
            if not inputs:
                continue
            status.parent.mkdir(parents=True, exist_ok=True)
            code, log = run_tool(ctx, self.id, ctx.layout.tools / tool,
                                 ['--exefs', ctx.temp / 'exefs', '--mods', mods.mods_dir(ctx.dest), '--patches', patches,
                                  '--out', ctx.temp / out, '--status', status])
            if code:
                status.unlink(missing_ok=True)
                skipped[warn] = log
            else:
                patches = ctx.temp / out
        args = ['--exefs', ctx.temp / 'exefs', '--out', gen]
        code, log = run_tool(ctx, self.id, ctx.layout.recomp / 'gen.py', args + ['--patches', patches], parse)
        if code and patches != plain:   # the lift does not take the mods' sets: without them
            for tool, out, status, warn in stages:
                status.unlink(missing_ok=True)
                skipped.setdefault(warn, log)
            code, log = run_tool(ctx, self.id, ctx.layout.recomp / 'gen.py', args + ['--patches', plain], parse)
        if code:
            raise InstallError('oot.lift_failed', code=code, log=str(log))
        for warn, wlog in skipped.items():
            rep.warning(warn, log=str(wlog))
        mods.write_built(ctx.dest, inputs)   # what this build was given: the launcher's Mods page reads it
        (gen / 'lift.done').write_text(ctx.keys[self.id] + '\n', encoding='utf-8')
        return {}


class Compile(Step):
    id = 'oot_compile'

    def key(self, ctx):   # (compile_base_key is this for the lift without mods: the two stay alike)
        layout = ctx.layout
        return fingerprint('oot-compile-2', ctx.keys['oot_lift'], tree_stamp(layout.root / 'runtime', layout.tools / 'build_game.py'),
                           tree_stamp(layout.pgo) if layout.pgo and layout.pgo.is_file() else 'no profile', cpu_level())

    def current(self, ctx, record):
        exe = ctx.temp / 'build' / ('rekarina' + EXE)
        return exe.is_file() and exe.stat().st_size == record.get('exe_size')

    def run(self, ctx):
        layout, rep, build = ctx.layout, ctx.reporter, ctx.temp / 'build'
        phase = ['compile']

        def parse(line):
            m = NINJA_LINE.match(line)
            if m:
                rep.progress(self.id, phase[0], int(m.group(1)), int(m.group(2)), 'objects')
            elif line.startswith('compile: ') or 'linking' in line.lower():
                phase[0] = 'link'
                rep.progress(self.id, 'link', force=True)

        rep.progress(self.id, 'compile', force=True)
        # With an execution profile the whole exe is compiled for it: the generated code at -O2 with the game's hot
        # paths known (the game's own frame is what limits the frame rate, and it is that code), the runtime linked as
        # one unit. The profile was trained on the C this lifter writes for this game version, so it applies here; a
        # function it does not know compiles without it. Without a profile the generated code stays at -O1 but for the
        # chunks with the game's hottest code (tools/data/hot-oot-v111.txt).
        pgo = layout.pgo if layout.pgo and layout.pgo.is_file() else None
        os.environ['REKARINA_RT_LTO'] = '1'
        os.environ.pop('REKARINA_PGO_RUNTIME_ONLY', None)
        hot = layout.tools / 'data' / 'hot-oot-v111.txt'
        if not pgo and hot.is_file():
            os.environ['REKARINA_HOT_OBJS'] = str(hot)
        else:
            os.environ.pop('REKARINA_HOT_OBJS', None)
        code, log = run_tool(ctx, self.id, layout.tools / 'build_game.py', [
            '--version', 'oot', '--gen', ctx.temp / 'gen', '--out', build, '--toolchain', layout.llvm, '--ninja', layout.ninja,
            '--deps', layout.deps, '--exefs', ctx.dest / 'exefs', '--no-romfs', *(['--pgo', pgo] if pgo else []),
            *(['--sysroot', layout.sysroot] if layout.sysroot else []),
            *(['--cpu', 'v2'] if cpu_level() == 'v2' else [])], parse)
        exe = build / ('rekarina' + EXE)
        if code or not exe.is_file():
            raise InstallError('oot.compile_failed', code=code, log=str(log))
        return {'exe_size': exe.stat().st_size}


class Components(Step):
    """The upscaler libraries for this machine's graphics card, beside the game (components.py: which, from where,
    under which license). Never fails the setup: without them the game runs in its base mode, the record says what
    is missing, and the next run (or the launcher's Retry) downloads only that."""

    id = 'oot_components'

    def key(self, ctx):
        # (our DLSS Frame Generation library, when the release carries it and the card runs DLSS Frame Generation:
        # releases 1.2.0 to 1.2.3 carried none, and a folder they set up gets it at its next update)
        gpu, shim = host_gpu(ctx), ctx.layout.shim
        ours = [(SHIM, shim)] if shim.is_file() and 'dlss' in components.groups_for(gpu) else []
        # (RenHDrum's library for any card, when the release carries it: 1.2.9 and older carried none, and a folder
        # they set up gets it at its next update)
        ours += [(RENHDRUM, ctx.layout.renhdrum)] if ctx.layout.renhdrum.is_file() else []
        return fingerprint('oot-components-1', components.pins_digest(), components.gpu_key(gpu), *ours)

    def current(self, ctx, record):
        return not record.get('failed') and all((ctx.dest / n).is_file() for n in record.get('files') or [])

    def run(self, ctx):
        rep, gpu = ctx.reporter, host_gpu(ctx)
        rep.progress(self.id, 'detect', force=True)
        rep.log(f'components: graphics card: {components.describe(gpu) or "not detected"}; '
                f'groups: {", ".join(components.groups_for(gpu))}')
        failed, files, ours = {}, [], []
        try:
            shim = ctx.layout.shim   # ours, bundled: frame generation loads Streamline through it
            if shim.is_file() and 'dlss' in components.groups_for(gpu):
                shutil.copy2(shim, ctx.dest / SHIM)
                ours = [SHIM]   # in the record's files: the step runs again when it is gone (current)
            if ctx.layout.renhdrum.is_file():   # ours too, bundled: the Vibration setting plays pads through it
                shutil.copy2(ctx.layout.renhdrum, ctx.dest / RENHDRUM)
                ours.append(RENHDRUM)
            result = components.install(ctx.dest, gpu, ctx.temp / 'components', log=rep.log,
                                        progress=lambda group, done, total: rep.progress(self.id, group, done, total, 'bytes'))
            failed = result['failed']
            files = result['installed'] + result['present'] + result['kept'] + ours
            for name in result['kept']:
                rep.log(f'components: {name} in the game folder is not the pinned version: kept as it is')
        except net.Cancelled:
            raise KeyboardInterrupt from None
        except Exception as e:   # anything else is ours to survive: the libraries are optional
            rep.log(f'components: {type(e).__name__}: {e}')
            failed = {'*': f'{type(e).__name__}: {e}'}
        shutil.rmtree(ctx.temp / 'components', ignore_errors=True)
        if failed:
            rep.warning('oot.components_failed', files=', '.join(n for n in failed if n != '*') or '-',
                        detail=next(iter(failed.values()))[:300])
        chosen = components.preset(gpu, components.usable(ctx.dest))
        return {'gpu': gpu.to_json() if gpu else None, 'groups': components.groups_for(gpu), 'files': files,
                'failed': sorted(failed), 'preset': {'id': chosen['id'], 'tex': chosen['tex']}}


class Prompts(Step):
    """The button prompt overlays and the fonts that carry them: a minute of drawing, redone when the drawing tools or
    the dump change. It is a step of its own because it once shared a key with the shader preparation, and a change
    to a prompt then ran the shaders again (half an hour or more)."""

    id = 'oot_prompts'

    TOOLS = ('make_oot_prompts.py', 'oot_hint_layout.py', 'button_icons.py', 'make_ext_font.py')

    def key(self, ctx):
        return fingerprint('oot-prompts-3', ctx.keys['oot_extract'], code_stamp(*(ctx.layout.tools / n for n in self.TOOLS)))

    def adopts(self, ctx, record):
        """A record of the key's form before 1.2.8 ('oot-prompts-2': the tools by their bytes) stands when those bytes
        are still the ones it was drawn with."""
        return record.get('key') == fingerprint('oot-prompts-2', ctx.keys['oot_extract'],
                                                tree_stamp(*(ctx.layout.tools / n for n in self.TOOLS)))

    def current(self, ctx, record):
        build = ctx.temp / 'build'
        return (build / 'romfs-patches-xbox' / 'manifest.tsv').is_file() and \
            all((build / f'shared-fonts-{d}' / '5-nintendo-extension.bfttf').is_file() for d, _ in STYLES)

    def run(self, ctx):
        layout, rep, build = ctx.layout, ctx.reporter, ctx.temp / 'build'
        rep.progress(self.id, 'prompts', force=True)
        # fontTools' own Python in place of its compiled modules (tools/fonttools_pure.py, the same fonts): the drawing
        # then runs on no native code but Python's, and a run that failed is run again (tool_steady)
        pure = {'REKARINA_FONTTOOLS_PURE': '1'}
        try:
            code, log = tool_steady(ctx, self.id, layout.tools / 'make_oot_prompts.py', ['--out', build, '--nup', ctx.nup],
                                    env=pure)
            if code:
                dump_damage(ctx, self.id, log, '--all')
                raise InstallError('oot.prompts_failed', code=code, log=str(log))
            for folder, style in STYLES:
                out = build / f'shared-fonts-{folder}'
                out.mkdir(parents=True, exist_ok=True)
                code, log = tool_steady(ctx, self.id, layout.tools / 'make_ext_font.py',
                                        [out / '5-nintendo-extension.bfttf', '--style', style], env=pure)
                if code:
                    raise InstallError('oot.prompts_failed', code=code, log=str(log))
        except Unstable as e:
            raise e.error() from None
        return {}


# Bump when a change to the runtime (or anything else the shader key does not read) alters what --prepare-shaders makes
# from the same inputs: every folder then prepares its shaders again. The compile key is not in the shader key on
# purpose (see Prepare.key), so this is the one hand switch.
SHADER_FORMAT = 1
# The tools the prepared shaders follow from, and the seed of variants play met (tools/data).
SHADER_TOOLS = ('oot_prepare_shaders.py', 'prepare_shaders.py', 'oot_romfs.py', 'shader_census.py')
SHADER_DATA = ('shader-answers-oot.bin.gz',)
# A folder that an earlier setup prepared has the old form of this step's key (it took in the compile key and the
# prompt tools, so it cannot be recomputed from what is on disk). The releases below prepared exactly what this setup
# would (every variant, the same seed and translator rules), so their record is taken over; any other release's is not.
LEGACY_PREPARED_BY = ('1.2.0', '1.2.1')


def shader_mode(ctx):
    """'all': every variant of the dump, in the foreground, for every setup run. 1.2.1 to 1.2.3 prepared the variants
    play met on a first install ('met') and left the rest to a background job of the launcher; with only those, play
    translated and built the rest in its frames (a player: "massive stuttering"), and the owner's call (2026-11-19) was
    that a setup prepares everything before play. A folder recorded 'met' therefore prepares the rest at its next setup
    run, from what it holds (seed_from_folder), and one that holds every variant under this translator prepares nothing
    (Prepare.current). --all-shaders and --shaders-later are still taken and change nothing."""
    return 'all'


def shader_case(ctx):
    """The shader step's case for the estimate (eta.py), from what it will do in this run: 'rerun' when it will be
    skipped (its record stands and its output is in the temp folder or the game folder, as Context.plan decides), else
    'translate' (every variant, translated: the case 'all' until 1.2.7 also built them in the driver, and its recorded
    times are not this step's). A folder that prepared before but has no prepared output where Prepare.current looks
    does that work in full, which a 'rerun' estimate (a minute) hid: the window counted down to a couple of minutes and
    stayed there."""
    rec = ctx.record('oot_prepare')
    step = Prepare()
    if rec and (rec.get('key') == step.key(ctx, 'all') or step.adopts(ctx, rec)) and step.current(ctx, rec):
        return 'rerun'
    return 'translate'


def shaders_only_met(state):
    """True for a folder whose shaders were prepared for the variants play met only (and has not prepared all since): its
    next setup run prepares the rest. A record of the old form (no 'shaders' field) prepared all."""
    rec = ((state or {}).get('steps') or {}).get('oot_prepare')
    return bool(rec) and rec.get('shaders') == 'met' and (state or {}).get('shaders') != 'all'


class Prepare(Step):
    """Every shader translated, and the driver's cache filled, before play (tools/oot_prepare_shaders.py)."""

    id = 'oot_prepare'

    def key(self, ctx, mode=None):
        """What the prepared shaders follow from, and nothing else: the dump, the translator's identity (the shader
        cache is keyed by it), the preparation tools and the seed, the mode. Not the compile key: the exe only carries
        the --prepare-shaders command, the shaders it makes follow from the translator and these inputs, and the
        package step builds the driver's cache again for the exe it copies (--warm-shaders), under the name the driver
        keys it by. A new exe therefore costs the compile and the package step, not half an hour of preparation."""
        layout = ctx.layout
        xid = layout.xlate.with_suffix('.id')
        # The dump's key is in ctx.keys once ctx.plan() ran. shader_mode also asks for this key before that (main's
        # estimate, which runs before run_steps plans): the extraction's key reads no other step's, so it is computed
        # here the way plan computes it.
        dump = ctx.keys['oot_extract'] if 'oot_extract' in ctx.keys else Extract().key(ctx)
        # (the identity file under the name it had before the translator was renamed)
        return fingerprint('oot-prepare-4', SHADER_FORMAT, mode or shader_mode(ctx), dump,
                           ('totk_xlate.id', xid) if xid.is_file() else tree_stamp(layout.xlate),
                           tree_stamp(*(layout.tools / n for n in SHADER_TOOLS), *(layout.tools / 'data' / n for n in SHADER_DATA)))

    PREPARING = '.shader-prepare-running'   # in the build folder from the start of run() until its end
    JOURNAL = 'shader-prepare-journal.bin'   # the runtime's, while the driver builds (vk_pipeline.c lb_warm_prepared)

    def current(self, ctx, record):
        """The output of a run that finished. --prepare-shaders writes shader-prepared.bin after the translation, before
        the driver's long phase, so a run cut short later leaves it: such a run also leaves PREPARING behind, and,
        cut short while the driver built (also by a setup from before PREPARING), the runtime's journal of that phase,
        which only a finished build removes.
        A setup unpacked again or a cleaned temp folder has none of that, while the game folder holds what the step
        would make (the record's key names every variant of this dump under this translator): then the step is done,
        and the package step, finding no shader output in the build folder, leaves the folder's as it is. 1.2.3 prepared
        the variants play met in that case and its package replaced the folder's pack of every variant (2 GB) with them
        (340 MB): play then translated and built the rest in its frames."""
        build = ctx.temp / 'build'
        if (build / 'shader-prepared.bin').is_file() and not (build / self.PREPARING).exists() and \
                not (build / self.JOURNAL).exists():
            return True
        nothing_here = not any((build / n).exists() for n in ('shader-prepared.bin', self.PREPARING, self.JOURNAL,
                                                              'shader-cache'))
        return nothing_here and record.get('shaders', 'all') == 'all' and folder_prepared(ctx)

    def adopts(self, ctx, record):
        """A record of the old key form is kept when its release prepared what this one would and the translator is the
        one that prepared it (the copy the step left in the build folder): a changed key form alone is no reason to
        prepare again."""
        if 'shaders' in record or record.get('release') not in LEGACY_PREPARED_BY:
            return False
        xid = ctx.layout.xlate.with_suffix('.id')
        for name in (xid.name, 'totk_xlate.id'):
            try:
                if xid.is_file() and (ctx.temp / 'build' / name).read_bytes() == xid.read_bytes():
                    return True
            except OSError:
                pass
        return False

    def adopted(self, ctx, record):
        record['shaders'] = 'all'   # what an old record means, kept in the new form
        ctx.state['shaders'] = 'all'

    def run(self, ctx):
        layout, rep, build = ctx.layout, ctx.reporter, ctx.temp / 'build'
        mode = shader_mode(ctx)
        # the translator beside the exe, where --prepare-shaders and the game look for it
        shutil.copy2(layout.xlate, build / layout.xlate.name)
        if layout.xlate.with_suffix('.id').is_file():
            shutil.copy2(layout.xlate.with_suffix('.id'), build / layout.xlate.with_suffix('.id').name)
        for lib in layout.libs:   # Linux: the game finds libSDL3.so.0 beside itself ($ORIGIN), here as in the package
            shutil.copy2(lib, build / lib.name)
        (build / self.PREPARING).write_text(mode + '\n', encoding='ascii')   # a run cut short leaves it (current)
        seed_from_folder(ctx, self.id, build)
        phase = ['seed']
        # Translation only (REKARINA_PREPARE_DRIVER=0, runtime/vk_pipeline.c vk_pipeline_prepare_shaders): the driver's
        # build of every prepared variant's stage library is left out of the setup. It held players up for 20 minutes
        # to 1.5 hours after the translation (a cold driver cache, an AMD card whose cache is keyed to the exe's path
        # so that it was built a second time in the game folder), while it only fills the driver's disk cache. The
        # game's own start builds what play recorded (its entry step, with its progress picture), and the launcher
        # builds the rest while the game is closed (background_main). Passed through the environment: the tools and
        # their key (SHADER_TOOLS) stay as they were, so no folder translates its shaders again for this.
        translate_only = {'REKARINA_PREPARE_DRIVER': '0'}

        def parse(line):
            m = TRANSLATED.search(line)
            if m:
                phase[0] = 'translate'
                rep.progress(self.id, 'translate', int(m.group(1)), int(m.group(2)), 'shaders')
            elif 'variants of' in line:
                phase[0] = 'translate'
                rep.progress(self.id, 'translate', force=True)
            elif LIBRARIES.search(line):   # (the driver's phase, which REKARINA_PREPARE_DRIVER=0 leaves out)
                m = LIBRARIES.search(line)
                phase[0] = 'libraries'
                rep.progress(self.id, 'libraries', int(m.group(1)), int(m.group(2)), 'shaders')

        rep.progress(self.id, 'seed', force=True)
        try:   # once more after an early failure that need not come back (tool_steady), not after a long run
            code, log = tool_steady(ctx, self.id, layout.tools / 'oot_prepare_shaders.py',
                                    [build, '--nup', ctx.nup, '--exefs', ctx.dest / 'exefs', '--exe-name', GAME_EXE,
                                     *(['--met-only'] if mode == 'met' else [])], parse, env=translate_only, tries=2,
                                    within=300)
        except Unstable as e:
            raise e.error() from None
        if code:
            dump_damage(ctx, self.id, log, '--shaders')
        if code or not (build / 'shader-prepared.bin').is_file():
            raise InstallError('oot.shaders_failed', code=code, log=str(log))
        # One shader-cache.pack instead of four files per shader (about 250,000 after the preparation): the game reads
        # the pack first, and with the loose files a start after the setup spent 7 s reading the shaders named by the
        # pipeline keys and 53 s reading their use masks, where a packed cache takes a fraction of a second; the
        # package step then copies one file instead of all of them.
        rep.progress(self.id, 'pack', force=True)
        code, log = run_tool(ctx, self.id, layout.tools / 'pack_shader_cache.py', [build / 'shader-cache'])
        if code:
            raise InstallError('oot.shaders_failed', code=code, log=str(log))
        (build / self.PREPARING).unlink(missing_ok=True)
        ctx.state['shaders'] = mode   # 'all' is kept through later updates (shader_mode)
        return {'shaders': mode}


# What a game folder keeps of a preparation beside its shader pack (tools/package_oot.py STATE), shader-prepared.bin aside:
# the pipeline cache, the pipelines and shaders the preparation and play recorded, the variants the driver failed on.
FOLDER_STATE = ('pipeline-cache.bin', 'pipeline-keys.bin', 'pipeline-keys-mesh.bin', 'shader-answers.bin',
                'shader-programs.bin', 'shader-prepare-skip.txt')


def folder_translator(ctx):
    """True when the game folder's shaders were made by this setup's translator: the identity file the package step
    wrote beside the pack (under its name from before the rename too) holds what this translator's does. The shader
    cache's entries are named by that identity, so another translator's are of no use."""
    xid = ctx.layout.xlate.with_suffix('.id')
    try:
        want = xid.read_bytes() if xid.is_file() else None
        return bool(want) and any((ctx.dest / n).is_file() and (ctx.dest / n).read_bytes() == want
                                  for n in ('rekarina_xlate.id', 'totk_xlate.id'))
    except OSError:
        return False


def folder_prepared(ctx):
    """The game folder holds a finished preparation under this setup's translator: its pack and shader-prepared.bin."""
    return folder_translator(ctx) and (ctx.dest / 'shader-prepared.bin').is_file() and \
        (ctx.dest / 'shader-cache' / 'shader-cache.pack').is_file()


def seed_from_folder(ctx, step_id, build):
    """Before a preparation: what the game folder holds of an earlier one under the same translator goes into the build
    folder first (its pack merged in, the files of FOLDER_STATE and the driver's nv-cache where the build folder has
    none), so the run translates and builds only what is missing, and what the package step copies back holds all that
    the folder held. A folder of 1.2.1-1.2.3 that prepared the variants play met prepares only the rest this way."""
    if not folder_translator(ctx):
        return
    cache = build / 'shader-cache'
    cache.mkdir(parents=True, exist_ok=True)
    pack = ctx.dest / 'shader-cache' / 'shader-cache.pack'
    if pack.is_file():
        ctx.reporter.progress(step_id, 'seed', force=True)
        code, log = run_tool(ctx, step_id, ctx.layout.tools / 'pack_shader_cache.py', [cache, '--merge-from', pack])
        if code:
            raise InstallError('oot.shaders_failed', code=code, log=str(log))
    for name in FOLDER_STATE:
        if (ctx.dest / name).is_file() and not (build / name).exists():
            shutil.copy2(ctx.dest / name, build / name)
    if (ctx.dest / 'nv-cache').is_dir() and not (build / 'nv-cache').exists():
        shutil.copytree(ctx.dest / 'nv-cache', build / 'nv-cache')
    ctx.reporter.log(f'shaders: what the game folder held under this translator was taken in first ({ctx.dest})')


LOOSE_SHADER_FILE = re.compile(r'[0-9a-f]{64}\.(spv|res\.json|use)$')   # tools/pack_shader_cache.py NAME, beside in/*.in


def has_loose_shader_files(cache):
    """Whether a shader cache holds files tools/pack_shader_cache.py would fold into its pack: stops at the first one."""
    try:
        with os.scandir(cache) as it:
            for e in it:
                if LOOSE_SHADER_FILE.match(e.name):
                    return True
        with os.scandir(cache / 'in') as it:
            return any(e.name.endswith('.in') for e in it)
    except OSError:
        return False


class Package(Step):
    id = 'oot_package'

    def key(self, ctx):
        opt = ctx.options
        # (the compile key only when mods changed the code: a folder without them keeps the key it had)
        built = ctx.keys['oot_compile']
        # (the texture step's tool and choice: a folder set up before it gets its textures prepared once)
        # (the compile key through compile_base_key: the shader key no longer holds it, and a new exe must be copied and
        # its driver cache warmed; the pack tool: a changed pack layout is converted here)
        return fingerprint('oot-package-4', ctx.keys['oot_prepare'], ctx.keys.get('oot_prompts', ''), compile_base_key(ctx),
                           ctx.layout.tools / 'package_oot.py', ctx.layout.tools / 'pack_shader_cache.py',
                           ctx.layout.tools / 'oot_prepare_textures.py', 'textures-no' if getattr(opt, 'no_textures', False) else '',
                           str(opt.dlls or ''), *([built] if built != compile_base_key(ctx) else []))

    def current(self, ctx, record):
        exe = ctx.dest / GAME_EXE
        return (exe.is_file() and exe.stat().st_size == record.get('exe_size') and
                any((ctx.dest / name).is_file() for name in (INI, OLD_INI)))

    def run(self, ctx):
        layout, rep, opt, build = ctx.layout, ctx.reporter, ctx.options, ctx.temp / 'build'
        # The build's shader cache in the pack layout of this setup (a no-op when it already is: the preparation packs
        # it, and a folder that skipped the preparation holds the pack of the release before).
        if (build / 'shader-cache').is_dir():
            rep.progress(self.id, 'pack', force=True)
            code, log = run_tool(ctx, self.id, layout.tools / 'pack_shader_cache.py', [build / 'shader-cache'])
            if code:
                raise InstallError('oot.package_failed', code=code, log=str(log))
        rep.progress(self.id, 'copy', force=True)
        if opt.dlls:   # the player's own DLSS / FSR / frame generation libraries go in with the build's
            for name in OPTIONAL_DLLS:
                if (Path(opt.dlls) / name).is_file() and not (build / name).is_file():
                    shutil.copy2(Path(opt.dlls) / name, build / name)
        # First settings for this machine's graphics card, for a rekarina.ini the tool writes new (an existing one is the
        # player's and stays). Chosen from what is in the folder now: a preset never switches on a library that
        # could not be downloaded.
        gpu = host_gpu(ctx)
        have = components.usable(ctx.dest) | (components.usable(build) if opt.dlls else set())
        chosen = components.preset(gpu, have)
        new_ini = not any((ctx.dest / name).is_file() for name in (INI, OLD_INI))
        preset_args = ['--preset-note', f'{components.describe(gpu) or "graphics card not detected"}: preset {chosen["id"]}']
        for k, v in chosen['settings'].items():
            preset_args += ['--set', f'{k}={v}']
        if fork.owns_install(layout.root):
            fork.mark_install(ctx.dest)
        code, log = run_tool(ctx, self.id, layout.tools / 'package_oot.py', [
            '--nup', ctx.nup, '--build', build, '--exefs', ctx.dest / 'exefs', '--xlate', layout.xlate, '--out', ctx.dest,
            '--prompts', opt.prompts or 'auto', '--lang', opt.lang or 'en-US', *preset_args,
            *[arg for lib in layout.libs for arg in ('--lib', lib)]])
        exe = ctx.dest / GAME_EXE
        if code or not exe.is_file():
            if code and exe.is_file():
                try:
                    open(exe, 'ab').close()
                except PermissionError:
                    raise InstallError('oot.exe_in_use', path=str(exe)) from None
            raise InstallError('oot.package_failed', code=code, log=str(log))
        # A folder set up by a release before 1.2.0 holds its shader cache as loose files (four per shader, about 250,000),
        # which the copy above leaves beside the new pack: the game would still read them at every start (seconds to
        # a minute of its entry step). Folded into the pack here, once; a folder without loose files is not touched.
        if has_loose_shader_files(ctx.dest / 'shader-cache'):
            rep.progress(self.id, 'pack', force=True)
            code, log = run_tool(ctx, self.id, layout.tools / 'pack_shader_cache.py', [ctx.dest / 'shader-cache'])
            if code:
                raise InstallError('oot.package_failed', code=code, log=str(log))
        # What the package step did after the copy until 1.2.7, the driver's build of every prepared variant in the
        # game folder (Rekarina.exe --warm-shaders) and the texture cache (tools/oot_prepare_textures.py), is the
        # launcher's to do now, in the background while the game is closed (background_main): on players' PCs the two
        # took 20 minutes to 1.5 hours and about 43 minutes, and the setup waited for them. The game's own start builds
        # the pipelines play recorded (its entry step, with its progress picture), and a texture the cache lacks is
        # made at its first use, as before 1.2.0.
        exe_sha256 = file_sha256(exe)
        plan_background(ctx, exe_sha256)
        write_rebuild_command(ctx)
        return {'exe_size': exe.stat().st_size, 'exe_sha256': exe_sha256,
                'ini_preset': chosen['id'] if new_ini else None}


REBUILD_BOOT = ('import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); '
                "runpy.run_module('installer.gui_oot', run_name='__main__', alter_sys=True)")


def rebuild_command(ctx):
    """The command that opens the launcher on this game folder and rebuilds it for its mods (gui_oot.py --rebuild):
    pythonw beside the setup's Python on Windows, so no console opens; in a checkout the Python that runs this."""
    layout = ctx.layout
    python = Path(sys.executable) if layout.dev else layout.python
    if WIN and python.with_name('pythonw.exe').is_file():
        python = python.with_name('pythonw.exe')
    return [str(python), '-c', REBUILD_BOOT, str(layout.root), '--root', str(layout.root), '--rebuild', str(ctx.dest)]


def write_rebuild_command(ctx):
    """build-cache/rebuild.txt in the game folder: rebuild_command, one argument per line. The game runs it at its
    start when a mod's code patches or code cheats are not in the build (runtime/cheats.c rebuild_if_needed)."""
    cache = ctx.dest / 'build-cache'
    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache / 'rebuild.txt.tmp'
    tmp.write_text('\n'.join(rebuild_command(ctx)) + '\n', encoding='utf-8', newline='\n')
    os.replace(tmp, cache / 'rebuild.txt')


# ---- the work the launcher does after the setup, while the game is closed
# <game folder>/build-cache/background.json: {"format": 1, "warm": entry, "textures": entry}, an entry being {"key": what
# the work follows from, "state": "pending" | "done" | "failed", "fails": n}. The package step writes it
# (plan_background); the launcher's background job (installer/shaderjob.py, gui_oot.py shader_wanted) runs
# `python -m installer.oot --dest <game folder> --background` (background_main) while the launcher is open and no game
# runs, at below-normal priority on all cores but one, and Play stops it within a moment (BACKGROUND_STOP).
#   warm      Rekarina.exe --warm-libraries: the game's entry step without its pipelines
#             (the records' shaders and their vertex output unions, as the game takes them), then every prepared
#             variant's stage library into the driver's disk cache (runtime/vk_pipeline.c lb_warm_prepared), so that a
#             pairing play has not recorded links from libraries the driver has (0.1 ms) instead of building them in a
#             frame (5-180 ms). The pipelines and pipeline-cache.bin are left to the game's start;
#             the mode keeps the file out of this process altogether (not read, not written).
#   textures  Rekarina.exe --scan-textures: every ASTC texture of the dump into
#             the texture cache, as play would write it (runtime/tex_prepare.c reads the archives itself).
# Neither is required: the game's entry step builds whatever is still missing before play, with its progress picture
# (runtime/vk_pipeline.c vk_pipeline_prewarm: the libraries in the game's own process; tex_prepare.c
# tex_prepare_entry_start: the walk through --scan-textures as its child); this only gives it a head start.
BACKGROUND_FILE = Path('build-cache') / 'background.json'
BACKGROUND_STOP = Path('build-cache') / 'background.stop'   # its appearance stops the job (REKARINA_WARM_STOP)
BACKGROUND_WORK = ('warm', 'textures')   # in this order: missing libraries cost stutter, a missing texture far less
BACKGROUND_TRIES = 2       # failures under one key before the job leaves the work alone (another key tries again)
BACKGROUND_GRACE = 1.5     # closing the launcher: seconds the job has before its tree is killed (with half a second
                           # more, stop_background)
BACKGROUND_STOP_WAIT = 20.0   # Play and Pause: seconds the launcher waits for the libraries in the driver's hands
                              # before it stops the job the hard way (a cold mesh library takes up to ~15 s)
BACKGROUND_CHILD_WAIT = BACKGROUND_STOP_WAIT - 2   # the job's own wait for its stopped child before it kills it: a
                              # build killed while it writes the driver's cache may leave that damaged
TEXTURE_SOURCES = ('tex_prepare.c', 'tex_astc.c', 'tex_astc_mt.c', 'tex_deswizzle.c', 'vk_texture.c', 'vk_astc.c',
                   'bc3_encode_spv.h', 'astc_decode_spv.h')   # what a texture cache file follows from (runtime/)
TEXTURE_ARCHIVES = re.compile(r'^textures: (\d+) of (\d+) archives')   # runtime/tex_prepare.c scan_rows


def background_state(dest):
    """The game folder's background.json as a dict ({} when there is none or it cannot be read)."""
    try:
        data = json.loads((Path(dest) / BACKGROUND_FILE).read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_background(dest, data):
    path = Path(dest) / BACKGROUND_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=1) + '\n', encoding='utf-8', newline='\n')
    os.replace(tmp, path)


def background_pending(dest):
    """The work the launcher still has to do for the game in dest, in order (BACKGROUND_WORK); [] when the folder has no
    game or nothing is left. Cheap: one small file read (the window asks once a second)."""
    if not (Path(dest) / GAME_EXE).is_file():
        return []
    data = background_state(dest)
    return [w for w in BACKGROUND_WORK if isinstance(data.get(w), dict) and data[w].get('state') == 'pending']


# The game's first start without the background work (the setup's finish screen says it): the entry step's recorded
# pipelines and the prepared variants' stage libraries on every thread, the textures beside them. Thread seconds an
# item with a cold driver cache (runtime/vk_pipeline.c: a graphics job 0.6 s; a library 0.47 s, the setup's driver
# phase: 56,402 in 1,650 s on 16 threads), and the texture walk's seconds on that machine.
FIRST_START_RECORDS = 10847          # the pipelines the release's seeds name
FIRST_START_PIPE_S = 0.6
FIRST_START_LIB_S = 0.47
FIRST_START_TEXTURES_S = 600


def first_start_estimate(dest, cores=None):
    """Seconds the next start of the game in dest should spend before play on what the background work has not done
    (background.json): 0 when nothing is left."""
    pending = background_pending(dest)
    if not pending:
        return 0.0
    cores = max(1, cores or os.cpu_count() or 8)
    try:
        variants = max(0, (os.path.getsize(Path(dest) / 'shader-prepared.bin') - 16) // 36) if 'warm' in pending else 0
    except OSError:
        variants = 0
    cpu = (FIRST_START_RECORDS * FIRST_START_PIPE_S + variants * FIRST_START_LIB_S) / cores
    return max(cpu, FIRST_START_TEXTURES_S if 'textures' in pending else 0.0)


def texture_key(layout, nup):
    """What the texture cache's files follow from: the runtime's texture code, the dump (path, size, time)."""
    try:
        st = os.stat(nup)
        dump = f'{Path(nup).resolve()}|{st.st_size}|{st.st_mtime_ns}'
    except (OSError, TypeError):
        dump = str(nup)
    runtime = layout.root / 'runtime'
    return fingerprint('bg-textures-2', dump, *(runtime / n for n in TEXTURE_SOURCES))


def plan_background(ctx, exe_sha256):
    """After the copy (Package.run): the background work this folder needs, written to background.json. A work whose key
    is the one done before stays done; any other is pending, with its failures counted from nothing."""
    dest = ctx.dest
    prepared = dest / 'shader-prepared.bin'
    want = {'warm': fingerprint('bg-warm-1', exe_sha256, prepared, ('rekarina_xlate.id', dest / 'rekarina_xlate.id'))
            if prepared.is_file() else None,
            'textures': None if getattr(ctx.options, 'no_textures', False) else texture_key(ctx.layout, ctx.nup)}
    old = background_state(dest)
    data = {'format': 1, 'nup': str(ctx.nup)}
    for work in BACKGROUND_WORK:
        before = old.get(work) if isinstance(old.get(work), dict) else {}
        if want[work] is None:
            data[work] = {'key': None, 'state': 'skipped', 'fails': 0}
        elif before.get('key') == want[work] and before.get('state') == 'done':
            data[work] = before
        else:
            data[work] = {'key': want[work], 'state': 'pending', 'fails': 0}
    (dest / BACKGROUND_STOP).unlink(missing_ok=True)
    save_background(dest, data)
    ctx.reporter.log('background work for the launcher: ' +
                     ', '.join(f'{w} {data[w]["state"]}' for w in BACKGROUND_WORK))


def stop_background(dest, proc=None, grace=BACKGROUND_GRACE + 0.5):
    """Stop the background job of the game in dest: BACKGROUND_STOP appears (the libraries' build ends after the ones in
    the driver's hands, runtime/vk_pipeline.c warm_stop_watch; background_main stops the texture tool), proc (the job,
    when the caller started it) gets grace seconds to end, then its whole tree is killed. A killed build leaves its
    journal behind; it names variants that did not crash the driver, so it goes (vk_pipeline.c skip_init would make
    them suspects). Returns True when the job ended by itself."""
    stop = Path(dest) / BACKGROUND_STOP
    try:
        stop.parent.mkdir(parents=True, exist_ok=True)
        stop.write_bytes(b'')
    except OSError:
        pass
    ended = proc is None
    if proc is not None:
        until = time.monotonic() + grace
        while proc.poll() is None and time.monotonic() < until:
            time.sleep(0.05)
        ended = proc.poll() is not None
        if not ended:
            from . import shaderjob
            tree = shaderjob.descendants(shaderjob.process_table(), proc.pid)
            kill_tree(proc)
            try:
                proc.wait(timeout=10)
            except (subprocess.TimeoutExpired, OSError):
                pass
            # Every process of the tree gone before the game starts: a killed Rekarina.exe still alive holds the
            # driver's cache file, and the game would then get an empty one of its own (vk_pipeline.c
            # vk_pipeline_prewarm). It was still listed right after taskkill returned.
            until = time.monotonic() + 5
            while tree & set(shaderjob.process_table()) and time.monotonic() < until:
                time.sleep(0.05)
            (Path(dest) / 'shader-prepare-journal.bin').unlink(missing_ok=True)
        stop.unlink(missing_ok=True)   # the job is gone: the file has done its work
    return ended


def wait_child(proc, stopped, poll=0.2):
    """Wait for proc in short slices, so that Ctrl+Break (KeyboardInterrupt) and the stop file reach this process:
    Popen.wait() without a timeout is one WaitForSingleObject that no signal handler interrupts on Windows (a setup's
    Cancel waited for the whole driver build that way). stopped(): True when the caller wants the child gone. Returns
    the exit code, or None when stopped() said so first (the child still runs)."""
    while True:
        try:
            return proc.wait(timeout=poll)
        except subprocess.TimeoutExpired:
            if stopped():
                return None


def background_main(args):
    """`--background`: the pending work of the game folder (background.json), in order, with events on stdout for the
    launcher (step "background", phases "libraries" and "textures" with their counts) and a log in the game folder's
    logs/launcher-background.log. Exit code 0 when everything pending is done, 3 when stopped, 1 when a work failed."""
    dest = args.dest.resolve()
    layout = find_layout(args.root, args.toolchain)
    logs = dest / 'logs'
    log = logs / 'launcher-background.log'
    try:
        if log.is_file() and log.stat().st_size > (1 << 20):
            log.replace(log.with_name('launcher-background.1.log'))
    except OSError:
        pass
    rep = Reporter(log_path=log)
    rep.log(f'---- {time.strftime("%Y-%m-%d %H:%M:%S")} background work, setup {layout.release}')
    stop = dest / BACKGROUND_STOP
    stop.unlink(missing_ok=True)
    data = background_state(dest)
    todo = background_pending(dest)
    rep.emit('start', release=layout.release, mode='background', dest=str(dest), log=str(log),
             steps=['background'], work=todo)
    exe = dest / GAME_EXE
    result = 'ok'
    for work in todo:
        if stop.exists():
            result = 'cancelled'
            break
        rep.step('background', 'started', work=work)
        t0 = time.monotonic()
        try:
            if work == 'warm':
                outcome = background_warm(rep, dest, exe, stop)
            else:
                outcome = background_textures(rep, layout, dest, exe, data.get('nup'), stop)
        except KeyboardInterrupt:
            outcome = 'stopped'
        except Exception as e:   # the work is optional: a failure is counted, not raised
            rep.log(traceback.format_exc())
            outcome = f'failed: {type(e).__name__}: {e}'
        data = background_state(dest)   # as it is now: another run may have written it meanwhile
        entry = data.get(work) if isinstance(data.get(work), dict) else None
        if entry is None:
            break
        if outcome == 'done':
            entry['state'] = 'done'
        elif outcome == 'stopped':
            result = 'cancelled'
        else:
            entry['fails'] = int(entry.get('fails') or 0) + 1
            entry['state'] = 'failed' if entry['fails'] >= BACKGROUND_TRIES else 'pending'
            entry['last_error'] = outcome[:300]
            result = 'failed'
        save_background(dest, data)
        rep.log(f'{work}: {outcome} in {time.monotonic() - t0:.0f} s')
        rep.step('background', 'done', work=work, outcome=outcome.split(':')[0], seconds=round(time.monotonic() - t0, 1))
        if result == 'cancelled':
            break
    stop.unlink(missing_ok=True)
    rep.emit('end', result=result, seconds=round(rep.elapsed(), 1))
    rep.close()
    return EXIT[result]


def background_child(rep, command, cwd, env, stop, parse, journal=None):
    """Run one process of the background work with its output parsed line by line (from a thread, so that the wait
    itself stays interruptible: wait_child). 'done', 'stopped' or 'failed: ...'."""
    import threading
    flags = subprocess.CREATE_NO_WINDOW if WIN else 0
    # (stdin a pipe held open: --warm-libraries and --scan-textures stop when it closes, this process gone)
    proc = subprocess.Popen(command, cwd=str(cwd), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1,
                            creationflags=flags, start_new_session=not WIN)
    tail = []

    def read():
        for line in proc.stdout:
            line = line.rstrip()
            tail.append(line)
            del tail[:-20]
            parse(line)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        code = wait_child(proc, stop.exists)
    except BaseException:
        kill_tree(proc)
        raise
    if code is None:   # the stop file: the libraries' build ends by itself (warm_stop_watch), anything else is killed
        try:
            code = proc.wait(timeout=BACKGROUND_CHILD_WAIT) if 'REKARINA_WARM_STOP' in env else None
        except subprocess.TimeoutExpired:
            code = None
        if code is None:
            kill_tree(proc)
            proc.wait(timeout=10)
            if journal is not None:   # a killed build's journal names innocent variants (stop_background)
                Path(journal).unlink(missing_ok=True)
            rep.log(f'{Path(command[0]).name}: stopped, killed after {BACKGROUND_CHILD_WAIT} s')
        else:
            rep.log(f'{Path(command[0]).name}: stopped by itself (exit code {code})')
        reader.join(timeout=2)
        return 'stopped'
    reader.join(timeout=5)
    rep.log(f'{Path(command[0]).name} {" ".join(command[1:])}: exit code {code}')
    for line in tail[-5:]:
        rep.log('  ' + line)
    if code == 3 and stop.exists():
        return 'stopped'
    return 'done' if code == 0 else f'failed: exit code {code}'


def background_warm(rep, dest, exe, stop):
    env = dict(os.environ, REKARINA_WARM_STOP=str(stop))

    def parse(line):
        m = LIBRARIES.search(line)
        if m:
            rep.progress('background', 'libraries', int(m.group(1)), int(m.group(2)), 'shaders')

    rep.progress('background', 'libraries', force=True)
    return background_child(rep, [str(exe), '--warm-libraries'], dest, env, stop, parse,
                            journal=dest / 'shader-prepare-journal.bin')


def background_textures(rep, layout, dest, exe, nup, stop):
    if not nup or not Path(nup).is_file():
        return f'failed: the dump {nup} is not there'

    def parse(line):
        m = TEXTURE_ARCHIVES.match(line)
        if m:
            rep.progress('background', 'textures', int(m.group(1)), int(m.group(2)), 'archives')

    rep.progress('background', 'textures', force=True)
    env = dict(os.environ, REKARINA_WARM_STOP=str(stop))
    return background_child(rep, [str(exe), '--scan-textures'], dest, env, stop, parse)


LAUNCHER_EXE = 'Rekarina-Launcher.exe'   # the setup folder's launcher (installer/update.py LAUNCHER)
LAUNCHER_LINK = 'Rekarina Launcher.lnk'   # beside the game executable
START_LINK = fork.START_SHORTCUT
DESKTOP_LINK = fork.DESKTOP_SHORTCUT


def shell_folder(csidl):
    """A folder of the player's as the shell has it (SHGetFolderPathW: a desktop moved to OneDrive is found there),
    None when Windows does not say."""
    if not WIN:
        return None
    import ctypes
    buf = ctypes.create_unicode_buffer(1024)
    if ctypes.windll.shell32.SHGetFolderPathW(None, csidl, None, 0, buf) != 0 or not buf.value:
        return None
    return Path(buf.value)


def start_menu_folder():
    """The player's own Start menu programs folder (CSIDL_PROGRAMS: no administrator needed)."""
    return shell_folder(0x02)


def desktop_folder():
    """The player's desktop (CSIDL_DESKTOPDIRECTORY)."""
    return shell_folder(0x10)


def write_shortcut(link, target, workdir, description):
    """A Windows shortcut written by the shell's own writer (WScript.Shell through PowerShell), so no module is needed.
    The paths reach PowerShell in environment variables, not in its command, so no character of theirs can change the
    command. The icon is the target's first one. (ok, what PowerShell said)."""
    script = ('$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:REKARINA_LNK); '
              '$s.TargetPath = $env:REKARINA_LNK_TARGET; $s.WorkingDirectory = $env:REKARINA_LNK_DIR; '
              '$s.IconLocation = $env:REKARINA_LNK_TARGET + ",0"; $s.Description = $env:REKARINA_LNK_TEXT; $s.Save()')
    env = dict(os.environ, REKARINA_LNK=str(link), REKARINA_LNK_TARGET=str(target), REKARINA_LNK_DIR=str(workdir),
               REKARINA_LNK_TEXT=description)
    proc = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', script],
                          env=env, capture_output=True, text=True, timeout=60,
                          creationflags=subprocess.CREATE_NO_WINDOW if WIN else 0)
    return proc.returncode == 0 and Path(link).is_file(), (proc.stderr or proc.stdout or '').strip()


def launcher_shortcuts(ctx):
    """The launcher one click from the game: "Rekarina Launcher" in the game folder, beside Rekarina.exe (the game
    alone, which players took for the launcher and then could not find where to update), "Rekarina" in the player's
    Start menu, and "Rekarina Launcher" on the desktop unless the form's desktop row says no (--no-desktop-shortcut).
    All point at the launcher of the setup that runs, in its own folder, so they are written again at every run: a
    setup downloaded anew (another <home>/setup/<version>) moves them to its launcher. Windows only (the Linux setup has
    no launcher executable); a source checkout has no launcher and gets none. A shortcut that cannot be written costs
    nothing but itself: it is logged. Returns the shortcuts written."""
    launcher = ctx.layout.root / LAUNCHER_EXE
    if not WIN or not launcher.is_file():
        return []
    made = []
    start, desk = start_menu_folder(), desktop_folder()
    if getattr(getattr(ctx, 'options', None), 'no_desktop_shortcut', False):
        desk = None
    for link in [ctx.dest / LAUNCHER_LINK] + ([start / START_LINK] if start else []) + ([desk / DESKTOP_LINK] if desk else []):
        try:
            link.parent.mkdir(parents=True, exist_ok=True)
            ok, said = write_shortcut(link, launcher, ctx.layout.root, 'Rekarina launcher: build, update and start the game')
        except (OSError, subprocess.SubprocessError) as e:
            ok, said = False, str(e)
        ctx.reporter.log(f'shortcut {link} -> {launcher}: ' + ('written' if ok else f'not written ({said})'))
        if ok:
            made.append(link)
    # The uninstaller beside the game and the Windows "Apps & features" entry (installer/uninstall.py), at every run:
    # they follow the game folder and the version
    if (ctx.dest / GAME_EXE).is_file():
        from . import uninstall
        try:
            uninstall.install(ctx.dest, getattr(ctx.layout, 'release', ''), setup=ctx.layout.root, log=ctx.reporter.log)
        except (OSError, subprocess.SubprocessError) as e:
            ctx.reporter.log(f'uninstaller: not set up ({e})')
    return made


def shortcut_target(link):
    """The program a Windows shortcut starts (the shell's own reader through PowerShell), None when it cannot be read."""
    script = '(New-Object -ComObject WScript.Shell).CreateShortcut($env:REKARINA_LNK).TargetPath'
    try:
        proc = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
                              env=dict(os.environ, REKARINA_LNK=str(link)), capture_output=True, text=True, timeout=60,
                              creationflags=subprocess.CREATE_NO_WINDOW if WIN else 0)
    except (OSError, subprocess.SubprocessError):
        return None
    out = proc.stdout.strip()
    return Path(out) if proc.returncode == 0 and out else None


def version_key(text):
    """'1.2.5' (or '1.2.5-dev') as (1, 2, 5); None for anything else."""
    m = re.match(r'(\d+(?:\.\d+)*)', (text or '').strip())
    return tuple(int(x) for x in m.group(1).split('.')) if m else None


def ensure_shortcuts(layout, dest, log=print):
    """The launcher's start (Windows, a release folder with its launcher, a game in dest): the shortcuts a setup run
    writes, kept pointing at a launcher that is there. "Rekarina Launcher" in the game folder and "Rekarina" in the
    Start menu are written when they are missing, when their launcher is gone, or when it is older than this one (a
    newer setup downloaded and started, whose setup has not run yet; a setup run cancelled after an update). The
    desktop one only when it exists: whether there is one was the player's choice. A shortcut to the same launcher or
    a newer one stays. The uninstaller's parts (installer/uninstall.py) are written when the game folder lacks its
    script. Returns the links written."""
    launcher = layout.root / LAUNCHER_EXE
    if not WIN or not launcher.is_file() or not (Path(dest) / GAME_EXE).is_file():
        return []
    dest = Path(dest)
    if not fork.owns_install(dest):
        return []
    start, desk = start_menu_folder(), desktop_folder()
    links = [dest / LAUNCHER_LINK] + ([start / START_LINK] if start else [])
    if desk and (desk / DESKTOP_LINK).is_file():
        links.append(desk / DESKTOP_LINK)
    mine = version_key(getattr(layout, 'release', ''))
    same = lambda a, b: os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))   # noqa: E731
    made = []
    for link in links:
        target = shortcut_target(link) if link.is_file() else None
        if target is not None and target.is_file():
            if same(target, launcher):
                continue
            version = target.parent / 'VERSION'
            theirs = version_key(version.read_text(encoding='utf-8', errors='replace')) if version.is_file() else None
            if theirs and mine and theirs >= mine:
                continue
        try:
            link.parent.mkdir(parents=True, exist_ok=True)
            ok, said = write_shortcut(link, launcher, layout.root, 'Rekarina launcher: build, update and start the game')
        except (OSError, subprocess.SubprocessError) as e:
            ok, said = False, str(e)
        log(f'shortcut {link} -> {launcher}: ' + ('written at the launcher\'s start' if ok else f'not written ({said})'))
        if ok:
            made.append(link)
    from . import uninstall
    if not (dest / uninstall.GAME_SCRIPT).is_file():
        try:
            uninstall.install(dest, getattr(layout, 'release', ''), setup=layout.root, log=log)
        except (OSError, subprocess.SubprocessError) as e:
            log(f'uninstaller: not set up ({e})')
    return made


def all_steps():
    return [Check(), Extract(), Lift(), Compile(), Components(), Prompts(), Prepare(), Package()]


class TimedReporter(Reporter):
    """The reporter that also keeps each finished step's duration for the estimates of the next run (eta.record)."""

    cases = {}

    def step(self, step, state, **fields):
        if state == 'done' and step in self.cases:
            eta.record(step, self.cases[step], fields.get('seconds'))
        return super().step(step, state, **fields)


class Parser(argparse.ArgumentParser):
    def error(self, message):
        Reporter().error('cli.usage', {'detail': message})
        sys.exit(2)


def cancel(signum, frame):
    raise KeyboardInterrupt


def main(argv=None):
    ap = Parser(prog='python -m installer.oot', description=__doc__.splitlines()[0])
    ap.add_argument('--game', type=Path, help="the dump: the .nup container or the folder that holds it")
    ap.add_argument('--dest', type=Path, required=True, help='install folder')
    ap.add_argument('--lang', help='game language, a Switch language code (en-US, ru, ja, de, ...)')
    ap.add_argument('--prompts', choices=('auto', 'xbox', 'nintendo', 'playstation', 'rekarina'),
                    help='button prompts of a new install (default auto: the controller or keyboard in use)')
    ap.add_argument('--dlls', type=Path, help='a folder with your DLSS / FSR 3.1 / frame generation libraries to put '
                                              'beside the game')
    ap.add_argument('--toolchain', type=Path, help='a folder with llvm/ and bin/ninja (a checkout has none of its own)')
    ap.add_argument('--root', type=Path, help='release folder or checkout (default: the folder above installer/)')
    ap.add_argument('--temp', type=Path, help='temp folder (default: temp/ of the release, build/install-temp of a checkout)')
    ap.add_argument('--until', help='stop after this step (oot_check ... oot_package): for tests')
    ap.add_argument('--all-shaders', action='store_true',
                    help='taken for older launchers: every setup run prepares every variant of every shader')
    ap.add_argument('--shaders-later', action='store_true',
                    help='taken for older launchers: changes nothing (every variant is prepared in the foreground)')
    ap.add_argument('--no-desktop-shortcut', action='store_true',
                    help='no "Rekarina Launcher" shortcut on the desktop (the game folder\'s and the Start menu\'s are '
                         'always written)')
    ap.add_argument('--no-textures', action='store_true',
                    help='do not prepare the textures for play (about 12 GiB on the disk; the game then prepares each one '
                         'at its first use)')
    ap.add_argument('--background', action='store_true',
                    help="the launcher's work after the setup for the game in --dest: the driver's shader libraries and "
                         'the texture cache (background_main); nothing else runs')
    args = ap.parse_args(argv)
    for name in ('SIGBREAK', 'SIGTERM'):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), cancel)
    if args.background:
        return background_main(args)
    rep, t0 = None, time.monotonic()
    try:
        dest = args.dest.resolve()
        layout = find_layout(args.root, args.toolchain)
        temp, moved = work_folder(layout.temp, args.temp, dest)
        log = temp / 'logs' / time.strftime('install-%Y%m%d-%H%M%S.log')
        rep = TimedReporter(log_path=log)
        if moved:
            rep.log(f'The disk of the setup has {moved} GiB free and the working files need about {NEED_TEMP >> 30}: '
                    f'they go into the game folder, {temp}')
        t0 = time.monotonic()
        try:
            state = statefile.load(dest)
        except statefile.StateError as e:
            rep.error(e.id, {'path': str(e.path)})
            rep.emit('end', result='failed', seconds=0)
            return 1
        if fork.owns_install(layout.root) and (dest / GAME_EXE).is_file() and not fork.owns_install(dest):
            rep.error('cli.usage', {'detail': 'Choose a separate game folder for this fork. Import your save on the Saves page.'})
            rep.emit('end', result='failed', seconds=0)
            return 1
        ctx = Context(layout, dest, temp, rep, args, state=state)
        steps = all_steps()
        if args.until:
            steps = steps[:[s.id for s in steps].index(args.until) + 1]
        rep.emit('start', release=layout.release, mode=ctx.mode, dev=layout.dev, dest=str(dest), log=str(log),
                 steps=[s.id for s in steps], game='oot')
        # How long each step should take (eta.py): "first" for a step this folder has not finished before, "rerun" for
        # one it has (its caches are there). The window counts down from these; each finished step's real time is kept
        # for the next run on this machine.
        # A step whose output left the temp folder (a setup unpacked again, a cleaned temp folder) does its first run's
        # work again, whatever the record says.
        done = (state or {}).get('steps', {})
        rep.cases = {s.id: 'rerun' if state and (s.always or (s.id in done and s.current(ctx, done[s.id]))) else 'first'
                     for s in steps}
        # the shader step by what it will prepare: nothing or every variant (its own case, 'all', with its own default
        # and its own recorded time)
        if 'oot_prepare' in rep.cases:
            rep.cases['oot_prepare'] = shader_case(ctx)
        rep.emit('eta', **eta.estimates(rep.cases))
        result = run_steps(steps, ctx)
        if result != 'cancelled' and (ctx.dest / GAME_EXE).is_file():
            launcher_shortcuts(ctx)   # at every run, so they follow the setup that ran last
    except Exception as e:
        # A failure outside the steps (run_steps reports its own): the traceback goes to the log, and the window gets
        # an error and the end of the run instead of a process that ended without either ("stopped unexpectedly").
        if rep is None:   # no log yet: stderr, which the window reads with stdout
            rep = TimedReporter()
            traceback.print_exc()
        rep.log(traceback.format_exc())
        rep.error('internal', {'exception': type(e).__name__, 'detail': str(e)})
        result = 'failed'
    rep.emit('end', result=result, seconds=round(time.monotonic() - t0, 1))
    rep.close()
    return EXIT[result]


if __name__ == '__main__':
    sys.exit(main())
