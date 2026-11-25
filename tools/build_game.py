"""Compile the generated C for one game version and link it with the runtime into build/<version>/rekarina.exe.

Writes build/<version>/build.ninja and runs MSYS2 UCRT64 Ninja on it. Ninja keeps the build incremental: a chunk is
recompiled only when it, a header it includes or its flags change.

On Linux the same steps make build/<version>/totk with the clang, lld and ninja on PATH (or the clang and lld in --toolchain DIR/bin): the
runtime's *_sdl.* and platform/linux.c take the place of its *_win.*, platform/win32.c and rekarina.rc, the libraries are
the system's (Vulkan, SDL3, Opus, libvpx, zstd; libstdc++ static), and the executable finds a libSDL3.so.0 beside it
first (rpath $ORIGIN). docs/building.md lists the packages.

    .venv\\Scripts\\python tools\\build_game.py                     # generated/v121 -> build/v121/rekarina.exe
    .venv\\Scripts\\python tools\\build_game.py --modules subsdk0   # one module (links if it has no calls into others)
    .venv\\Scripts\\python tools\\build_game.py --jobs 8
    .venv\\Scripts\\python tools\\build_game.py --out build/v121-t2   # a separate build directory
    .venv\\Scripts\\python tools\\build_game.py --toolchain toolchain/llvm-mingw-<version>/llvm-mingw-<version>-ucrt-x86_64 \\
        --out build/v121-llvmmingw                                    # llvm-mingw instead of MSYS2 UCRT64

Every runtime/*.c is compiled. build/<version>/modules.c carries the layout, the linked function tables, the
expected SHA-256 of each ExeFS file (tools/data/expected-<display version>.json) and the default game paths
(--exefs, --romfs, --profile), which rekarina.exe uses unless its own command line names others. --exefs and --romfs
default to extracted/<version> in the work folder: the [work] dir of config.local.toml, where tools/extract.py writes
them, else build/. tools/romfs_index.py writes build/<version>/romfs-index.bin and romfs_config.h from the RomFS
directory's index.json (runtime/romfs.c reads the index next to rekarina.exe). Without that index.json the build stops:
the exe would stop at start. --no-romfs builds it anyway.

Private builds reuse the main build's objects. When --out differs from build/<version>, the main build's generated
objects are up to date (checked with a dry run of its build.ninja) and every header they include from runtime/ is
byte-identical in --runtime (the list comes from the main build's Ninja deps log), build_game.py hardlinks those
objects into <out>/obj and compiles only runtime/*.c. A hardlink is a snapshot: clang and Ninja replace an object by
renaming a new file over it, so a later main rebuild leaves the linked copies alone. Otherwise (a header differs, a
main object is missing or stale, another --gen) it compiles everything and prints one line saying why. --full-build
skips the check.

Object manifest. A flipped bit in an object written by an earlier build stays forever, because Ninja rebuilds an object
only when its inputs change, and forks hardlink the object into every private build. So each build dir keeps
obj/objects.sha256 (SHA-256, size and mtime of every object it compiled). After Ninja finishes and before the link,
build_game.py hashes the objects, compares each one whose size and mtime match its manifest entry, and records the
objects Ninja just rebuilt (new mtime). A mismatch deletes the object, Ninja recompiles it, and the build prints the
summary line. A fork checks the hardlinked generated objects against build/<version>/obj/objects.sha256 (read-only; a
bad shared object stops the fork and asks for a main build run, which recompiles it and breaks the shared inode).

Content stamps. Ninja decides by file times, and a release folder's sources carry the release archive's times, which
can be older than the objects an earlier release's setup left in temp/. So obj/inputs.sha256 records what each object
and the exe were made from (sources, headers from Ninja's deps log, command, compiler version), and an output whose
inputs changed is deleted before Ninja runs (the comment at INPUTS; tests/build_content_test.py).

On Windows every compile starts when the memory free at that moment holds its estimate (MemoryGate: the compilers
Ninja starts are held suspended until then), so programs the player opens or closes mid-compile lower or raise the
number running. Elsewhere, or with REKARINA_BUILD_MEM_GB, the jobs share a budget fixed at the start: the memory
available then less MEM_RESERVE for the rest of the PC, at most MEM_BUDGET. The chunks that are one giant function need
gigabytes each and run in a pool of their own, largest first (chunk_memory, plan_jobs); the others take LIGHT_MEM.
Every compile and the link run below normal priority. Each phase reports its wall time and the peak memory of its
processes (Windows job object accounting: committed private bytes).

--ninja PATH names the Ninja to run; without MSYS2 the bin/ninja.exe of the release folder (tools/make_release.py) is used.
--toolchain DIR builds with an llvm-mingw tree (DIR/bin/clang.exe) instead of MSYS2 UCRT64 clang. Ninja still comes
from MSYS2. The libraries and headers MSYS2 supplies (Vulkan, SDL3, Opus, libvpx, VMA) then come from --deps DIR
(default toolchain/deps-mingw: include/, lib/ with the import libraries and libvpx.a, bin/ with the DLLs), and the C++
runtime is libc++ from llvm-mingw. Such a build never reuses another build's generated objects.

Split build. --make-runtime-lib compiles only runtime/ into <out>/librekarina.a (what CI prebuilds). --runtime-lib PATH
then compiles only the generated C and the install-time files (modules.c and INSTALL_SIDE, which bake the player's
game hashes and version in) and links them with that library. The library holds no game-derived data.

Standard library only.
"""
import argparse
import ctypes
import ctypes.wintypes as wt   # the job-object structures below; importing it works on every OS
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import check_isa
import romfs_index
from build_shaders import stale as stale_shaders
from check_content import GENERATED_MARKER

WIN = os.name == 'nt'
if not WIN:
    import resource
ROOT = Path(__file__).resolve().parent.parent
# In a git worktree (tools/worktree.py) build/ and generated/ are junctions into the main checkout, and resolve() follows
# them. MAIN_ROOT is where they land, so resolved paths still print and compare relative to a root.
MAIN_ROOT = (ROOT / 'build').resolve().parent
UCRT = Path(r'C:\msys64\ucrt64\bin')
EXE = 'rekarina.exe' if WIN else 'rekarina'   # rekarina.exe before 2026-11-15


def on_path(name):
    """name on PATH, else a path that does not exist (the tool check below names it)."""
    found = shutil.which(name)
    return Path(found) if found else Path('/usr/bin') / name


CLANG = UCRT / 'clang.exe' if WIN else on_path('clang')
NINJA = UCRT / 'ninja.exe' if WIN else on_path('ninja')
SDL3_DLLS = [UCRT / 'SDL3.dll', UCRT / 'libiconv-2.dll', UCRT / 'libopus-0.dll'] if WIN else []

# Flags for generated code as LAS Recomp builds it (Makefile GENFLAGS, Windows), plus -march for the FP/SIMD lowering.
# --cpu v2 replaces the level for CPUs without AVX2/FMA; runtime/fma.c then supplies the correctly rounded fma the
# lowering needs (runtime/simd.h).
GEN_FLAGS = ['-O1', '-fno-strict-aliasing', '-Wno-return-type', '-Wno-unused-label', '-fvisibility=hidden',
             '-march=x86-64-v3']
RT_FLAGS = ['-O2', '-g', '-Wall', '-fno-strict-aliasing', '-march=x86-64-v3']


def set_cpu(cpu):
    """Target x86-64-v3 (the default) or x86-64-v2. RT_CPU_LEVEL lets runtime/main.c's self-test check the same
    instruction set the build was compiled for."""
    march = f'-march=x86-64-{cpu}'
    GEN_FLAGS[GEN_FLAGS.index('-march=x86-64-v3')] = march
    RT_FLAGS[RT_FLAGS.index('-march=x86-64-v3')] = march
    RT_FLAGS.append(f'-DRT_CPU_LEVEL={3 if cpu == "v3" else 2}')


# Profile-guided optimisation (--pgo). Both modes compile the generated code at -O2, the level the profile's inlining and
# hot/cold layout work at. A static function's profile name carries its source path by default; the base name alone
# keeps the names equal between the training build and a build from another tree (a worktree, an install).
PGO_NAMES = ['-mllvm', '-static-func-full-module-prefix=false']
PGO_INPUT = None   # the .profdata of --pgo FILE: every compile depends on it, so a new profile rebuilds them


def set_lto(cpu):
    """REKARINA_RT_LTO=1: ThinLTO across the runtime (the generated code stays out of it: its objects are not rebuilt).
    Not in a --cpu v2 build: its runtime objects must be machine code for tools/check_isa.py to read (with ThinLTO they
    are LLVM bitcode, llvm-objdump refused the first of them and every setup on a CPU without AVX2 stopped there)."""
    if os.environ.get('REKARINA_RT_LTO') == '1' and cpu != 'v2':
        RT_FLAGS.append('-flto=thin')
        LINK_FLAGS.extend(['-flto=thin', '-O2'])


def set_pgo(pgo, out):
    """--pgo gen: instrument the generated code and the runtime; rekarina.exe adds its counts to
    out/pgo/default_<id>.profraw every minute and at exit (runtime/platform win32.c), across runs. --pgo FILE: compile
    with that merged profile (tools/pgo.py merge). Functions the profile never saw run are optimised for size."""
    global PGO_INPUT
    # REKARINA_PGO_RUNTIME_ONLY=1: only the host runtime takes the profile; the generated code keeps its flags and objects
    rt_only = os.environ.get('REKARINA_PGO_RUNTIME_ONLY') == '1'
    if not rt_only:
        GEN_FLAGS[GEN_FLAGS.index('-O1')] = '-O2'
    if pgo == 'gen':
        # absolute: the instrumented exe resolves the directory against its own working directory
        flags = [f'-fprofile-generate={cmd_arg((out / "pgo").resolve().as_posix())}']
        RT_FLAGS.append('-DREKARINA_PGO_GEN')
        LINK_FLAGS.append('-fprofile-generate')   # links clang_rt.profile
    else:
        PGO_INPUT = Path(pgo).resolve()
        if not PGO_INPUT.is_file():
            sys.exit(f'--pgo {pgo}: no such profile')
        # unprofiled: chunks the training never entered; out-of-date and backend-plugin: functions whose code changed
        # since the training, which then compile without a profile
        flags = [f'-fprofile-use={cmd_path(PGO_INPUT)}', '-Wno-profile-instr-unprofiled',
                 '-Wno-profile-instr-out-of-date', '-Wno-backend-plugin']
    if not rt_only:
        GEN_FLAGS.extend(flags + PGO_NAMES)
    RT_FLAGS.extend(flags + PGO_NAMES)
if WIN:
    # _WIN32_WINNT: Windows 10, the port's minimum. MSYS2's headers default to it, llvm-mingw's (the release toolchain)
    # to Windows 7, which hides the per-monitor DPI functions gfx_window_win.c and dpi_probe_win.c call.
    RT_FLAGS.append('-D_WIN32_WINNT=0x0A00')
    # -Xlinker passes the map path as one argument; -Wl, would split it at any comma in the build dir path
    LINK_FLAGS = ['-fuse-ld=lld', '-Wl,--stack,16777216', '-Xlinker', '"--Map=$builddir/rekarina.map"']   # the map feeds tools/prof_report.py
    LINK_LIBS = ['-lpsapi', '-lbcrypt', '-lvulkan-1', '-luser32', '-lgdi32', '-ldwmapi', '-limm32', '-lwinmm', '-lole32', '-lshell32', '-luuid', '-lSDL3', '-lopus', '-l:libvpx.a', '-l:libzstd.a', '-l:libstdc++.a', '-l:libwinpthread.a']   # Vulkan back end; libvpx static (movies); libzstd static (resolution.c)
    VMA_INCLUDE = UCRT.parent / 'include' / 'vma'   # vk_mem_alloc.h; MSYS2 package vulkan-memory-allocator
else:
    # -no-pie: the lifted modules' function tables (hundreds of thousands of pointers) need no load-time relocation.
    # $ORIGIN: a libSDL3.so.0 next to totk comes before the system's (the release's portable folder).
    LINK_FLAGS = ['-fuse-ld=lld', '-no-pie', '-Wl,-z,noexecstack', "'-Wl,-rpath,$$ORIGIN'",
                  '"-Wl,-Map=$builddir/rekarina.map"']
    LINK_LIBS = ['-lvulkan', '-lSDL3', '-lopus', '-lvpx', '-lzstd', '-l:libstdc++.a', '-lm', '-lpthread', '-ldl']
    VMA_INCLUDE = Path('/usr/include')   # vk_mem_alloc.h; Arch (AUR) vulkan-memory-allocator, Debian libvulkan-memory-allocator-dev
# Without a system vk_mem_alloc.h, the header the Windows release pins (tools/make_release.py, VMA 3.4.0) is fetched once
# into toolchain/vma: one MIT header (THIRD_PARTY_NOTICES.md), checked against its hash.
VMA_FETCH = ('https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/v3.4.0/include/vk_mem_alloc.h',
             '8487b7995ad3b263eb73bc5b9a77d71aa69b6bef5d58a715c02d2663afd81f1a')
VMA_LOCAL = ROOT / 'toolchain' / 'vma'


def vma_include():
    """The directory holding vk_mem_alloc.h: the system's, else toolchain/vma, fetched there on first use."""
    if (VMA_INCLUDE / 'vk_mem_alloc.h').is_file():
        return VMA_INCLUDE
    dst = VMA_LOCAL / 'vk_mem_alloc.h'
    if not dst.is_file():
        url, sha = VMA_FETCH
        print(f'build_game: no {VMA_INCLUDE / "vk_mem_alloc.h"}; fetching VMA 3.4.0 into {VMA_LOCAL}')
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
        except OSError as e:
            sys.exit(f'build_game: fetching {url} failed ({e}); install VMA or put vk_mem_alloc.h in {VMA_LOCAL}')
        if hashlib.sha256(data).hexdigest() != sha:
            sys.exit(f'build_game: {url} does not match the pinned hash')
        VMA_LOCAL.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
    return VMA_LOCAL


# --toolchain: an llvm-mingw tree. The libraries, headers and DLLs MSYS2 supplies come from the deps directory instead.
LLVM_LINK_LIBS = ['-lpsapi', '-lbcrypt', '-luser32', '-lgdi32', '-ldwmapi', '-limm32', '-lwinmm', '-lole32', '-lshell32', '-luuid', '-lvulkan-1', '-lSDL3', '-lopus',
                  '-l:libvpx.a', '-l:libdav1d.a', '-l:libzstd.a', '-l:libc++.a', '-l:libc++abi.a', '-l:libunwind.a', '-l:libwinpthread.a']
DEFAULT_DEPS = ROOT / 'toolchain' / 'deps-mingw'
# Runtime sources compiled per install, not in the library: romfs.c includes romfs_config.h, patches.c and ov_zstd.c
# the patches.h of the player's game version (its patch page is the page after that version's main; ov_zstd.c's host
# functions sit on it). mouse_direct.c, oot_fps.c and wide_ui.c also register host callbacks relative to that page.
INSTALL_SIDE = {'romfs.c', 'patches.c', 'ov_zstd.c', 'mouse_direct.c', 'oot_fps.c', 'wide_ui.c'}
LIB_NAME = 'librekarina.a'
AR = UCRT / 'llvm-ar.exe' if WIN else on_path('llvm-ar')
WINDRES = UCRT / 'windres.exe'   # runtime/rekarina.rc: the manifest of rekarina.exe
TOOLCHAIN_BIN = None   # llvm-mingw bin directory, set by main() for --toolchain
DEPS = None            # the deps directory of an llvm-mingw build
# tools/data/expected-<display version>.json for each build version name.
DISPLAY_VERSIONS = {'v0': '1.0.0', 'v110': '1.1.0', 'v112': '1.1.2', 'v121': '1.2.1', 'v143': '1.4.3'}
# Other games on the same runtime (Rekarina): build version name -> (expected-hash file stem, rt_game_version, the
# defines that select the game's behaviour in runtime/). rt_game_version is kept far from TotK's 110..143 so no RT_V*
# test ever matches another game. GAME_NS2: a Switch 2 game (NVN2, SDK 20+ host behaviour); GAME_OOT: OoT.
OTHER_GAMES = {'oot': ('expected-oot-v1.1.1', 9111, ['-DGAME_NS2', '-DGAME_OOT'])}

GiB = 1 << 30
# Memory of the compile jobs. Peak commit of one compile (2026-11-20, OoT's release generated C with the release's clang
# at -O2 and the profile, one compile at a time, measured with run_measured's job accounting): a chunk of many functions
# 100-330 MiB (main_0177.c, 1,541 functions in 3.8 MB of C: 218 MiB in 5 s), main_table.c 480 MiB, a runtime file with
# ThinLTO under 100 MiB; but a chunk that is one giant function (the lifter gives each function of a megabyte or so of C
# a chunk of its own) far more: main_0270.c (one 1.0 MB function) 657 MiB, main_0240.c (1.3 MB) 1,940 MiB in 102 s,
# main_0272.c (2.2 MB) 2,022 MiB, main_0282.c (3.9 MB) 2,801 MiB in 128 s. The old flat 768 MiB a job let twelve
# compiles start on a 16 GB PC with 12 threads, and it froze there three times out of three: the giants ran together.
# So each chunk's need is estimated from the manifest's chunk_info (bytes of C per function), the giants run in a pool
# of their own (at most so many at once), and the budget is the memory AVAILABLE when the build starts, less MEM_RESERVE
# for Windows, the desktop and the setup's window (compile_budget, plan_jobs).
# The giant line sits at 0.9 MB of one function: below it the one-function chunks are cheap (0.27-0.9 MB: 81-434 MiB,
# the most main_0258.c, 0.77 MB in 15 s), above it are all the heavy ones (from 1.2 MB, 1.9-2.8 GiB). A few above the line are cheap too
# (main_0185.c, 1.6 MB of many small branches, 2 s): the pool only makes them wait their turn. A lower line (0.25 MB
# was tried) puts over a hundred chunks into the giants' pool one at a time, and a 16 GB PC's build took twice as long.
LIGHT_MEM = 448 << 20          # any compile but a giant one's: the most measured under the giant line, 434 MiB
                               # (640 until 1.2.7: a 16 GB PC of a player ran four jobs and compiled for 57 minutes)
GIANT_FN_BYTES = 900_000       # a chunk whose functions average this much C or more is a giant one
GIANT_MEM_BASE = 1792 << 20    # a giant chunk's estimate: this plus GIANT_MEM_PER_MB for each MB of its function, with
GIANT_MEM_PER_MB = 358 << 20   # headroom (1.3 MB -> 2.2 GiB against 1.9 measured, 3.9 MB -> 3.1 GiB against 2.7)
MEM_RESERVE = 3072 << 20       # kept free for the rest of the PC (the memory available already leaves out what
                               # runs; 3.5 GiB until 1.2.7)
MEM_BUDGET = 50 * GiB          # at most, for all compile jobs together


def below_root(p):
    """Resolved p relative to ROOT, else to MAIN_ROOT (the same relative path through the worktree's junctions), else
    absolute."""
    p = Path(p).resolve()
    for root in (ROOT, MAIN_ROOT):
        if p.is_relative_to(root):
            return p.relative_to(root)
    return p


def ninja_path(p):
    """A path for a build line: relative to the repo root where possible, forward slashes, Ninja-escaped."""
    return below_root(p).as_posix().replace('$', '$$').replace(' ', '$ ').replace(':', '$:')


def flag_path(p):
    return below_root(p).as_posix()


def cmd_arg(s):
    """s for a Ninja command line: Ninja-escaped, and quoted when it holds a space (a user or folder name), as clang and
    lld split their Windows command line at unquoted spaces."""
    s = s.replace('$', '$$')
    return f'"{s}"' if ' ' in s else s


def cmd_path(p):
    return cmd_arg(flag_path(p))


def read_layout(gen):
    """[(name, base, size)] from layout.json (same format as recomp/layout.py read(), without its NSO dependency)."""
    data = json.loads((gen / 'layout.json').read_text())
    return [(m['name'], int(m['base'], 16), int(m['size'], 16)) for m in data]


def chunk_files(gen, mod):
    """The C files of one module: the chunks listed in <mod>/manifest.json, then <mod>_table.c."""
    d = gen / mod
    data = json.loads((d / 'manifest.json').read_text())
    entries = data if isinstance(data, list) else data.get('chunks', data.get('files'))
    if entries is None:
        sys.exit(f'{d / "manifest.json"}: no "chunks" or "files" list')
    names = []
    for e in entries:
        name = e if isinstance(e, str) else e.get('file', e.get('path', e.get('name')))
        names.append(Path(name).name)
    table = f'{mod}_table.c'
    if table not in names:
        names.append(table)
    files = [d / n for n in names]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        sys.exit(f'{mod}: {len(missing)} files listed in manifest.json are missing, first {missing[0]}')
    return files


def write_if_changed(path, text):
    """Keep the mtime of an unchanged file so Ninja does not rebuild from it."""
    if path.exists() and path.read_text(encoding='utf-8', errors='replace') == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')   # Ninja and clang read UTF-8; paths may be non-ASCII


def copy_romfs_patches(version, out):
    """rekarina.exe loads romfs-patches beside romfs-index.bin (runtime/romfs.c overlay_init). tools/make_romfs_patches.py
    writes them into build/<version> only, so a build in another directory ran without the 60 fps overlay (the quick
    menus' captured background turned black). Copies them when their manifest differs; romfs-patches-xbox,
    romfs-patches-xbox-label and romfs-patches-pc (the same with Xbox or keyboard and mouse button prompts,
    make_romfs_patches.py --prompts) the same way when they were made."""
    for name in ('romfs-patches', 'romfs-patches-xbox', 'romfs-patches-xbox-label', 'romfs-patches-pc'):
        src = ROOT / 'build' / version / name
        dst = out / name
        if src.resolve() == dst.resolve() or not (src / 'manifest.tsv').is_file():
            continue
        if (dst / 'manifest.tsv').is_file() and (dst / 'manifest.tsv').read_bytes() == (src / 'manifest.tsv').read_bytes():
            continue
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        print(f'romfs patches: copied {name} from {src}')


def work_dir():
    """The work folder: the [work] dir of config.local.toml (relative to the checkout), where tools/extract.py writes
    extracted/; build/ without the file or the key."""
    cfg = ROOT / 'config.local.toml'
    if not cfg.is_file():
        return ROOT / 'build'
    import tomllib
    return ROOT / tomllib.loads(cfg.read_text(encoding='utf-8')).get('work', {}).get('dir', 'build')


def c_string(s):
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


RUNTIME = ROOT / 'runtime'   # --runtime: a copy of runtime/ lets a task build without touching the shared tree


def other_os(name):
    """A runtime file for another OS: Win32 ones (*_win.*, platform/win32.c, rekarina.rc) off Windows, the SDL window's
    (*_sdl.*) and platform/linux.c on Windows."""
    stem = name.rsplit('.', 1)[0]
    if WIN:
        return stem.endswith('_sdl') or name == 'linux.c'
    return stem.endswith('_win') or name in ('win32.c', 'rekarina.rc')


def runtime_sources(skip=()):
    """Every runtime/*.c, *.cpp, platform/*.c, *.rc, the in-game menu (ui/*.cpp) and Dear ImGui of this OS except the
    names in skip, so a task adds a file without editing this script."""
    pats = ('*.c', '*.cpp', 'platform/*.c', '*.rc', 'ui/*.cpp', 'third_party/imgui/*.cpp',
            'third_party/imgui/backends/*.cpp')
    return sorted(p for pat in pats for p in RUNTIME.glob(pat) if p.name not in skip and not other_os(p.name))


def runtime_object(out, src):
    """obj/runtime/<path below runtime>.o, so platform/win32.c cannot collide with a top-level win32.c."""
    src = Path(src)
    rel = src.relative_to(RUNTIME) if src.is_relative_to(RUNTIME) else Path(src.name)
    return out / 'obj' / 'runtime' / rel.with_suffix('.o')


def module_glue(layout, linked, hashes, paths, version):
    """C for build/<version>/modules.c: the layout table, the linked modules' function tables (dispatch.h), the
    expected ExeFS hashes, the game version and the default game paths (host.h)."""
    out = [f'// Module glue written by tools/build_game.py; do not edit. {GENERATED_MARKER}',
           '#include "dispatch.h"', '#include "host.h"', '#include "layout.h"', '']
    for name, base, size in layout:
        u = name.upper()
        out.append(f'_Static_assert(MOD_{u}_BASE == {base:#x}ull && MOD_{u}_SIZE == {size:#x}ull, '
                   f'"layout.h and layout.json disagree on {name}");')
    out += ['', 'const RtLayoutEntry rt_layout[] = {']
    out += [f'    {{"{n}", MOD_{n.upper()}_BASE, MOD_{n.upper()}_SIZE}},' for n, _, _ in layout]
    out += ['};', f'const int rt_layout_count = {len(layout)};', '']
    for name in linked:
        out += [f'extern const uint64_t {name}_fn_addrs[];', f'extern const GuestFn {name}_fn_ptrs[];',
                f'extern const uint64_t {name}_fn_count;']
    out += ['', 'int rt_linked_modules(RtModule *out, int max) {', '    int n = 0;']
    for name in linked:
        u = name.upper()
        out.append(f'    if (n < max) out[n++] = (RtModule){{"{name}", MOD_{u}_BASE, MOD_{u}_SIZE, {name}_fn_addrs, '
                   f'{name}_fn_ptrs, {name}_fn_count}};')
    out += ['    return n;', '}', '']
    out += ['const RtFileHash rt_exefs_sha256[] = {']
    out += [f'    {{{c_string(f)}, {c_string(h)}}},' for f, h in sorted(hashes.items())]
    out += ['};', f'const int rt_exefs_sha256_count = {len(hashes)};', '']
    for name in ('exefs', 'romfs', 'profile'):
        out.append(f'const char rt_default_{name}[] = {c_string(paths[name].as_posix())};')
    number = int(DISPLAY_VERSIONS[version].replace('.', '')) if version in DISPLAY_VERSIONS else 0   # 1.4.3 -> 143
    if version in OTHER_GAMES:
        number = OTHER_GAMES[version][1]
    out += [f'const int rt_game_version = {number};', f'const char rt_port_version[] = {c_string(port_version())};', '']
    return '\n'.join(out)


def write_version_header(out):
    """out/rekarina_version.h for the version resource of runtime/rekarina.rc: the first three numbers of VERSION (1.1.0,
    1.2.0-rc1 -> 1,2,0,0). Rewritten only when it changes, so the resource is not compiled again on every build."""
    nums = [int(n) for n in re.findall(r'\d+', port_version().split('-')[0].split('+')[0])[:3]]
    nums += [0] * (4 - len(nums))
    text = (f'#define RK_VERSION_NUM {nums[0]},{nums[1]},{nums[2]},0\n'
            f'#define RK_VERSION_STR "{nums[0]}.{nums[1]}.{nums[2]}"\n')
    path = Path(out) / 'rekarina_version.h'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.read_bytes() != text.encode():
        path.write_bytes(text.encode())


def port_version():
    """VERSION at the root: the repository's semver, or in a release folder the release string (make_release.py
    version)."""
    path = ROOT / 'VERSION'
    return path.read_text(encoding='utf-8').strip() if path.is_file() else 'unknown'


def ninja_file(gen, out, mods, pool_depth, skip=(), prebuilt=False, runtime_lib=None, make_lib=False, edges=None,
               giants=None, giant_depth=1):
    """build.ninja text, and the object lists per module. With prebuilt the generated objects already exist in
    out/obj and are link inputs without a compile edge. runtime_lib: a prebuilt library replaces the runtime objects
    except INSTALL_SIDE. make_lib: build only that library, from the runtime objects except INSTALL_SIDE. edges, a
    dict, receives {output: (its command without $in and $out, [its inputs])} for every compile and the link (the
    content stamps below). giants: {generated C file: estimated peak} of the giant chunks (chunk_memory), compiled in
    a pool of giant_depth and listed first, largest first, so they start early and never run more at once."""
    if edges is None:
        edges = {}
    giants = giants or {}
    inc = f'-I {cmd_path(RUNTIME)} -I {cmd_path(gen)}'
    vma = f'{cmd_path(DEPS / "include")} -I {cmd_path(DEPS / "include" / "vma")}' if DEPS else cmd_path(vma_include())
    imgui = cmd_path(RUNTIME / 'third_party' / 'imgui')
    rt_inc = f'{inc} -I {cmd_path(out)} -I {vma} -I {imgui}'   # romfs_config.h, vk_mem_alloc.h (+ deps headers), imgui.h
    libs = [f'-L {cmd_path(DEPS / "lib")}'] + (LLVM_LINK_LIBS if WIN else LINK_LIBS) if DEPS else LINK_LIBS
    # A compiler cache (sccache) may front clang: every lift rewrites the generated chunks with the same contents, so
    # the cached objects save the compile step on the next build.
    cache = f'{cmd_arg(os.environ["SCCACHE"])} ' if os.environ.get('SCCACHE') else ''
    lines = [
        '# Written by tools/build_game.py; do not edit.',
        'ninja_required_version = 1.10',
        f'builddir = {ninja_path(out)}',
        f'cc = {cache}{cmd_arg(CLANG.as_posix())}',
        f'genflags = {" ".join(GEN_FLAGS)} {inc}',
        f'rtflags = {" ".join(RT_FLAGS)} {rt_inc}',
        f'ldflags = {" ".join(LINK_FLAGS)}',
        f'libs = {" ".join(libs)}',
        '',
        'pool gen',
        f'  depth = {pool_depth}',
        '',
        'pool gen_giant',
        f'  depth = {max(1, giant_depth)}',
        '',
        'rule cc_gen',
        '  command = $cc $genflags -MD -MF $out.d -c $in -o $out',
        '  deps = gcc',
        '  depfile = $out.d',
        '  description = CC $in',
        '',
        'rule cc_rt',
        '  command = $cc $rtflags -MD -MF $out.d -c $in -o $out',
        '  deps = gcc',
        '  depfile = $out.d',
        '  description = CC $in',
        '',
        'rule rc',
        # the game defines pick the icon (rekarina.rc)
        f'  command = {cmd_arg(WINDRES.as_posix())} -I {cmd_path(RUNTIME)} -I {cmd_path(out)} '
        f'{" ".join(f for f in RT_FLAGS if f.startswith("-DGAME_"))} -O coff $in -o $out',
        '  description = RC $in',
        '',
        'rule link',
        '  command = $cc $ldflags @$out.rsp $libs -o $out',
        '  rspfile = $out.rsp',
        '  rspfile_content = $in_newline',
        '  description = LINK $out',
        '',
    ]
    objs = {}
    pgo_dep = f' | {ninja_path(PGO_INPUT)}' if PGO_INPUT else ''
    # REKARINA_HOT_OBJS=<file>: one object name per line (first word), compiled at -O2 where the rest stays at -O1. Most of the
    # game's time is in a small share of the chunks, and -O2 across all of them takes hours.
    hot_list = os.environ.get('REKARINA_HOT_OBJS')
    hot_objs = {l.split()[0] for l in open(hot_list) if l.strip()} if hot_list else set()
    hot_flags = ' '.join('-O2' if f == '-O1' else f for f in GEN_FLAGS) + ' ' + inc
    # REKARINA_COLD_OBJS=<file>: one object name per line (first word), compiled at -O1 where the rest is at -O2 (a full
    # profile): a measurement of the giant chunks the profile never entered (main_0282.c, 3.9 MB of one function, takes
    # two minutes and 2.8 GiB at -O2); off unless set.
    cold_list = os.environ.get('REKARINA_COLD_OBJS')
    cold_objs = {l.split()[0] for l in open(cold_list) if l.strip()} if cold_list else set()
    cold_flags = ' '.join('-O1' if f == '-O2' else f for f in GEN_FLAGS) + ' ' + inc
    gen_pgo = PGO_INPUT if PGO_INPUT and os.environ.get('REKARINA_PGO_RUNTIME_ONLY') != '1' else None
    gen_lines = {}   # generated C file: its build lines
    for mod, files in mods.items():
        objs[mod] = []
        for f in files:
            o = out / 'obj' / mod / (f.stem + '.o')
            if not prebuilt:
                el = [f'build {ninja_path(o)}: cc_gen {ninja_path(f)}{"" if os.environ.get("REKARINA_PGO_RUNTIME_ONLY") == "1" else pgo_dep}']
                hot = Path(o).name in hot_objs
                cold = not hot and Path(o).name in cold_objs
                if hot:   # REKARINA_HOT_OBJS: the chunks where the profile finds the game's time
                    el.append(f'  genflags = {hot_flags}')
                elif cold:   # REKARINA_COLD_OBJS
                    el.append(f'  genflags = {cold_flags}')
                el.append('  pool = gen_giant' if f in giants else '  pool = gen')
                gen_lines[f] = el
                flags = hot_flags if hot else cold_flags if cold else ' '.join(GEN_FLAGS) + ' ' + inc
                edges[o] = (f'{cache}{cmd_arg(CLANG.as_posix())} {flags}',
                            [f] + ([gen_pgo] if gen_pgo else []))
            objs[mod].append(o)
    # the giants first, largest first: Ninja starts edges of equal weight in the order of the file
    for f in sorted(gen_lines, key=lambda f: -giants.get(f, 0)):
        lines += gen_lines[f]
    rt = []
    sources = runtime_sources(skip)
    if make_lib:
        sources = [p for p in sources if p.name not in INSTALL_SIDE]
    elif runtime_lib:
        sources = [p for p in sources if p.name in INSTALL_SIDE] + [out / 'modules.c']
    else:
        sources += [out / 'modules.c']
    for src in sources:
        o = runtime_object(out, src)
        if src.suffix == '.rc':   # windres writes no depfile: the files rekarina.rc embeds are listed by hand
            write_version_header(out)
            embeds = [RUNTIME / 'rekarina.manifest', RUNTIME / 'totk.ico', RUNTIME / 'rekarina.ico', out / 'rekarina_version.h']
            lines.append(f'build {ninja_path(o)}: rc {ninja_path(src)} | ' + ' '.join(ninja_path(e) for e in embeds))
            edges[o] = (f'{WINDRES.as_posix()} {" ".join(f for f in RT_FLAGS if f.startswith("-DGAME_"))}', [src] + embeds)
        else:
            lines.append(f'build {ninja_path(o)}: cc_rt {ninja_path(src)}{pgo_dep}')
            edges[o] = (f'{cache}{cmd_arg(CLANG.as_posix())} {" ".join(RT_FLAGS)} {rt_inc}',
                        [src] + ([PGO_INPUT] if PGO_INPUT else []))
        rt.append(o)
    objs['runtime'] = rt
    every = [o for v in objs.values() for o in v]
    exe = out / EXE
    if make_lib:
        exe = out / LIB_NAME
        lines += ['', 'rule ar', f'  command = {cmd_arg(AR.as_posix())} rcs $out $in', '  description = AR $out', '', f'build {ninja_path(exe)}: ar ' + ' '.join(ninja_path(o) for o in every),
                  'build objects: phony ' + ' '.join(ninja_path(o) for o in every), f'default {ninja_path(exe)}', '']
        return '\n'.join(lines), objs, exe
    rule = 'link'
    if runtime_lib:
        # Whole archive: the runtime's weak defaults (nvn_procs.c, nvn_api_core.c, vk_sync.c) would otherwise satisfy
        # references before the linker pulls the members with the strong definitions, which would stay out of rekarina.exe.
        rule = 'link_lib'
        lines += ['', 'rule link_lib',
                  f'  command = $cc $ldflags @$out.rsp -Wl,--whole-archive {cmd_path(runtime_lib)} -Wl,--no-whole-archive $libs -o $out',
                  '  rspfile = $out.rsp', '  rspfile_content = $in_newline', '  description = LINK $out']
    # the link's own inputs besides the objects: the runtime library and the static and import libraries of deps/lib
    lib_files = sorted(p for p in (DEPS / 'lib').glob('*') if p.is_file()) if DEPS else []
    edges[exe] = (f'{rule} {" ".join(LINK_FLAGS)} {" ".join(libs)} {len(every)} objects',
                  ([runtime_lib] if runtime_lib else []) + lib_files)
    lines += ['', f'build {ninja_path(exe)}: {rule} ' + ' '.join(ninja_path(o) for o in every) +
              (f' | {ninja_path(runtime_lib)}' if runtime_lib else ''),
              'build objects: phony ' + ' '.join(ninja_path(o) for o in every), f'default {ninja_path(exe)}', '']
    return '\n'.join(lines), objs, exe


def main_objects(gen, main_out, rt_dir, mods):
    """({relative object path: the main build's object}, '') when the main build's generated objects can stand in for
    ours, else (None, why). That holds when the main build.ninja compiles the same sources with the same flags, its
    objects are up to date, and every header they include from runtime/ is byte-identical in rt_dir."""
    main_ninja = main_out / 'build.ninja'
    if not main_ninja.is_file():
        return None, f'{flag_path(main_ninja)} not found'
    if gen != (ROOT / 'generated' / main_out.name).resolve():
        return None, 'generated C directory is not the main one'
    lines = main_ninja.read_text(encoding='utf-8').splitlines()
    sources = dict(l[6:].split(': cc_gen ') for l in lines if l.startswith('build ') and ': cc_gen ' in l)
    flags = next((l for l in lines if l.startswith('genflags = ')), '')
    if not flags.startswith('genflags = ' + ' '.join(GEN_FLAGS) + ' -I '):
        return None, 'main build uses other compile flags'
    pairs = {}
    for mod, files in mods.items():
        for f in files:
            o = Path('obj') / mod / (f.stem + '.o')
            theirs = main_out / o
            if sources.get(ninja_path(theirs)) != ninja_path(f):
                return None, f'main build has no edge for {f.name}'
            if not theirs.is_file():
                return None, f'main object {flag_path(theirs)} is missing'
            pairs[o] = theirs
    # Ninja runs in the main checkout: the main build.ninja is relative to it, and the worktree's runtime/ files carry newer
    # mtimes that would make every object look stale. Header equality is checked byte for byte below.
    env = dict(os.environ, PATH=str(UCRT) + os.pathsep + os.environ.get('PATH', '')) if WIN else dict(os.environ)
    targets = [ninja_path(t) for t in pairs.values()]
    for i in range(0, len(targets), 200):
        r = subprocess.run([str(NINJA), '-f', str(main_ninja), '-n', '-d', 'explain'] + targets[i:i + 200], cwd=MAIN_ROOT,
                           env=env, capture_output=True, text=True)
        if r.returncode or 'no work to do' not in r.stdout:
            return None, f'main generated objects are stale ({flag_path(main_out)} needs a build)'
    r = subprocess.run([str(NINJA), '-f', str(main_ninja), '-t', 'deps'], cwd=MAIN_ROOT, env=env, capture_output=True,
                       text=True)
    ours = {flag_path(t) for t in pairs.values()}
    headers, cur = set(), False
    for l in r.stdout.splitlines():
        if l.startswith(' '):
            if cur:
                headers.add(l.strip())
        else:
            cur = l.split(': #deps')[0] in ours
    main_rt = MAIN_ROOT / 'runtime'
    for h in sorted(headers):
        p = Path(h) if Path(h).is_absolute() else MAIN_ROOT / h
        try:
            rel = p.resolve().relative_to(main_rt)
        except ValueError:
            continue
        mine = rt_dir / rel
        if not mine.is_file() or mine.read_bytes() != p.read_bytes():
            return None, f'header {rel.as_posix()} differs in {flag_path(rt_dir)}'
    return pairs, ''


def link_objects(out, pairs):
    """Hardlink the main build's objects into out. Clang and Ninja replace an object by renaming a new file over it,
    so a link keeps the old bytes after a main rebuild. Copies where linking fails."""
    for o, theirs in pairs.items():
        dst = out / o
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if os.path.samefile(dst, theirs):
                continue
            dst.unlink()
        try:
            os.link(theirs, dst)
        except OSError:
            shutil.copy2(theirs, dst)


def unlink_shared(out, mods):
    """Drop generated objects still hardlinked to another build, so the next compile writes a new file."""
    for mod, files in mods.items():
        for f in files:
            o = out / 'obj' / mod / (f.stem + '.o')
            if o.exists() and o.stat().st_nlink > 1:
                o.unlink()


MANIFEST = 'objects.sha256'


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb', buffering=0) as f:
        buf = bytearray(1 << 20)
        while n := f.readinto(buf):
            h.update(memoryview(buf)[:n])
    return h.hexdigest()


def read_manifest(path):
    """{object path below obj/: (sha256, size, mtime_ns)}; empty when the file is missing."""
    man = {}
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            sha, size, mtime, rel = line.split(' ', 3)
            man[rel] = (sha, int(size), int(mtime))
    return man


def write_manifest(path, man):
    write_if_changed(path, ''.join(f'{sha} {size} {mtime} {rel}\n' for rel, (sha, size, mtime) in sorted(man.items())))


def check_objects(out, objects, man, record):
    """Compare objects with the manifest man. An object whose size and mtime match its entry is hashed and must equal
    it; any other object was just (re)built, so with record its hash enters man. Returns (verified, bad, unrecorded):
    the objects with a wrong hash, and with record False the ones without a current entry."""
    verified, bad, unrecorded = 0, [], []
    for o in objects:
        rel = o.relative_to(out / 'obj').as_posix()
        st = o.stat()
        e = man.get(rel)
        if e and e[1] == st.st_size and e[2] == st.st_mtime_ns:
            if file_sha256(o) == e[0]:
                verified += 1
            else:
                bad.append(o)
        elif record:
            man[rel] = (file_sha256(o), st.st_size, st.st_mtime_ns)
            verified += 1
        else:
            unrecorded.append(o)
    return verified, bad, unrecorded


def ninja_build(base, targets, env):
    for i in range(0, len(targets), 200):
        code = subprocess.run(base + targets[i:i + 200], cwd=ROOT, env=env).returncode
        if code:
            sys.exit(code)


def verify_objects(out, objs, borrowed, main_out, pairs, base, env):
    """Verify every object before the link; see the module docstring. borrowed: the hardlinked generated objects of
    main_out (checked against its manifest, never recorded); the others belong to out."""
    t0 = time.monotonic()
    owned = [o for mod, v in objs.items() for o in v if not (borrowed and mod != 'runtime')]
    shared = [o for mod, v in objs.items() for o in v if borrowed and mod != 'runtime']
    man_path = out / 'obj' / MANIFEST
    man = read_manifest(man_path)
    verified, bad, _ = check_objects(out, owned, man, True)
    recompiled = 0
    if bad:
        for o in bad:
            print(f'build_game: object {flag_path(o)} does not match its hash; deleting it to recompile')
            o.unlink()
        ninja_build(base, [ninja_path(o) for o in bad], env)
        recompiled = len(bad)
        verified += check_objects(out, bad, man, True)[0]
    write_manifest(man_path, man)
    if shared:
        main_man = read_manifest(main_out / 'obj' / MANIFEST)
        theirs = [pairs[o.relative_to(out)] for o in shared]
        v, bad, unrecorded = check_objects(main_out, theirs, main_man, False)
        verified += v
        if bad:
            for b in bad:
                print(f'build_game: shared object {flag_path(b)} does not match its hash in '
                      f'{flag_path(main_out / "obj" / MANIFEST)}', file=sys.stderr)
            sys.exit(f'build_game: run tools/build_game.py for {flag_path(main_out)} (it recompiles them), then rebuild '
                     f'{flag_path(out)}')
        if unrecorded:
            print(f'build_game: warning: {len(unrecorded)} shared objects have no current hash in '
                  f'{flag_path(main_out / "obj" / MANIFEST)}; build {flag_path(main_out)} once to record them')
    print(f'objects verified: {verified}, recompiled after a hash mismatch: {recompiled} ({time.monotonic() - t0:.1f} s)')


# ---- Content stamps: what each object was compiled from
#
# Ninja compiles an object again when an input has a newer file time than the object. In a checkout every file is
# written on this machine and that holds. In a release folder it does not: the sources come out of the release archive
# with the archive's file times, which are DOS local times without a time zone, written in the maintainer's zone and read
# in the player's. installer/update.py stage() sets each unpacked file's time from the archive with time.mktime (the
# player's zone), the launcher's swap (installer/launcher/rekarina.c apply_update) renames the staged entries into place
# and keeps those times, and a folder unpacked by hand gets whatever the unpacker sets. temp/ (the generated C and the
# objects of the last build) is kept across updates. So a source of a newer release can carry an older time than the
# object an earlier release's setup compiled from its old version, and Ninja keeps the old object: the update from 1.2.2
# to 1.2.4 compiled a new main.o (a header of it came out newer) beside the old report.o and the link failed on
# fg_guard_crashed (2026-11-20).
#
# So the content decides as well. obj/inputs.sha256 holds, per output, a SHA-256 of what it was made from: the compiler's
# version, the edge's command, and the content of its source, of every header Ninja's deps log lists for it (the depfile
# clang wrote at its last compile) and of the profile or other files the edge names. Before Ninja runs, an output whose
# record differs from its inputs now is deleted, and Ninja makes it again. A build folder without the record (made before
# this check) compiles all its runtime objects again (about a minute) and keeps its generated objects: their C is written
# on this machine by the lift, so Ninja's times hold for it, and the runtime headers they include (cpu.h, simd.h,
# simd_fp.h) are the same in every release from 1.2.0 to 1.2.4. The record is written after the compile and the link.
INPUTS = 'inputs.sha256'
_content = {}


def norm_path(p):
    """p (relative to ROOT, or absolute) as one key per file."""
    return os.path.normcase(os.path.normpath(os.path.join(ROOT, p)))


def output_key(p):
    """An output's key in the deps log: its real path (a worktree's build/ is a junction into the main checkout)."""
    return os.path.normcase(os.path.realpath(os.path.join(ROOT, p)))


def content_sha(p):
    n = norm_path(p)
    if n not in _content:
        try:
            _content[n] = file_sha256(n)
        except OSError:
            _content[n] = 'missing'
    return _content[n]


def ninja_deps(ninja_file_path, env):
    """{output_key of an output: the inputs Ninja's deps log lists for it} (empty without a log)."""
    r = subprocess.run([str(NINJA), '-f', str(ninja_file_path), '-t', 'deps'], cwd=ROOT, env=env, capture_output=True,
                       text=True, errors='replace')
    deps, cur = {}, None
    for line in r.stdout.splitlines():
        if not line.strip():
            cur = None
        elif line[0] not in ' \t':
            cur = deps.setdefault(output_key(line.split(': #deps')[0]), [])
        elif cur is not None:
            cur.append(line.strip())
    return deps


def compiler_id():
    """clang's version line: objects of another compiler (a release that brings a new toolchain) are made again."""
    try:
        r = subprocess.run([str(CLANG), '--version'], capture_output=True, text=True, timeout=120)
        return (r.stdout.splitlines() or ['unknown'])[0]
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


def inputs_key(ident, cmd, files):
    root = os.path.normcase(str(ROOT))
    h = hashlib.sha256(f'{ident}\n{cmd}\n'.encode())
    for n in sorted({norm_path(f) for f in files}):
        rel = os.path.relpath(n, root) if n.startswith(root + os.sep) else n
        h.update(f'{rel.replace(os.sep, "/")} {content_sha(n)}\n'.encode())
    return h.hexdigest()


def read_inputs(path):
    """{output relative to the build folder: key}, None when the file is missing."""
    if not path.is_file():
        return None
    out = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        key, _, rel = line.partition(' ')
        if rel:
            out[rel] = key
    return out


def current_keys(out, edges, env, outputs):
    deps, ident = ninja_deps(out / 'build.ninja', env), compiler_id()
    return {o.relative_to(out).as_posix(): inputs_key(ident, edges[o][0], edges[o][1] + deps.get(output_key(o), []))
            for o in outputs}


def drop_stale_by_content(out, edges, env):
    """Delete each existing output whose recorded inputs differ from its inputs now (see above), before Ninja runs."""
    t0 = time.monotonic()
    old = read_inputs(out / 'obj' / INPUTS)
    present = [o for o in edges if o.is_file()]
    if old is None:
        stale = [o for o in present if o.relative_to(out).as_posix().startswith('obj/runtime/')]
        why = 'no record of what they were compiled from: a build folder from before this check'
    else:
        now = current_keys(out, edges, env, present)
        stale = [o for o in present if old.get(o.relative_to(out).as_posix()) != now[o.relative_to(out).as_posix()]]
        why = 'their source, a header, the flags or the compiler changed since they were made'
    for o in stale:
        o.unlink()
    if stale:
        names = ', '.join(o.name for o in stale[:8]) + (', ...' if len(stale) > 8 else '')
        print(f'build_game: {len(stale)} outputs made again by content ({why}): {names}')
    print(f'content check: {len(present)} outputs, {len(stale)} to make again ({time.monotonic() - t0:.1f} s)', flush=True)


def record_inputs(out, edges, env, outputs):
    """Write the record for outputs (made and current now), keeping the other entries."""
    path = out / 'obj' / INPUTS
    keys = read_inputs(path) or {}
    keys.update(current_keys(out, edges, env, [o for o in outputs if o.is_file()]))
    write_if_changed(path, ''.join(f'{k} {rel}\n' for rel, k in sorted(keys.items())))


# ---- Windows job object: peak memory of a process tree
class IoCounters(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ('r_ops', 'w_ops', 'o_ops', 'r_bytes', 'w_bytes', 'o_bytes')]


class BasicLimits(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong), ('PerJobUserTimeLimit', ctypes.c_longlong),
                ('LimitFlags', wt.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wt.DWORD),
                ('Affinity', ctypes.c_size_t), ('PriorityClass', wt.DWORD), ('SchedulingClass', wt.DWORD)]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [('BasicLimitInformation', BasicLimits), ('IoInfo', IoCounters),
                ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]


JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JOB_OBJECT_LIMIT_PRIORITY_CLASS = 0x20
BELOW_NORMAL_PRIORITY_CLASS = 0x4000
CREATE_SUSPENDED = 0x4


# ---- the memory gate: each compiler started when memory holds it
# Ninja runs a job per thread and knows nothing of memory. The gate (Windows) holds every compiler Ninja starts
# suspended at its start and lets it run when its estimate (chunk_memory) fits what the PC has free NOW: the smaller of
# the physical memory available and the commit headroom (the commit limit less the commit charge), less what the running
# compiles will still take (each one's estimate less what it holds) and GATE_RESERVE. A browser or a game opened
# mid-compile so holds the next compiles back, and closing it lets more run; with none running the oldest waiting one
# runs whatever the memory, so the compile always goes on. Smaller compiles may pass the oldest waiting one for
# GATE_PATIENCE seconds, then none does, so a giant chunk is not starved by light ones; the giants keep their own pool
# besides (ninja_file). Ninja's commands stay as they were: a changed command line would compile every object again on
# a player's update.
GATE_RESERVE = 1536 << 20      # kept free besides the estimates (the gate checks again before every start)
GATE_PATIENCE = 30.0           # seconds the oldest waiting compile lets smaller ones pass it
GATE_TICK = 0.2                # seconds between checks
GATE_SAY = 10.0                # seconds between the gate's status lines
# the source of a compile's command line, quoted (a folder name with a space) or not
GATE_SOURCE = re.compile(r'\s-c\s+(?:"([^"]+?\.(?:c|cc|cpp))"|([^"\s]+?\.(?:c|cc|cpp)))(?=\s|$)', re.IGNORECASE)


def gate_pick(waiting, growth, free, running, oldest_wait, reserve=GATE_RESERVE, patience=GATE_PATIENCE):
    """Indexes of the waiting compiles (their estimates, oldest first) to start now. free: the memory free now (the
    smaller of the physical memory available and the commit headroom); growth: what the running compiles will still
    take; running: how many run; oldest_wait: seconds the oldest one has waited."""
    room = free - reserve - growth
    start = []
    for i, est in enumerate(waiting):
        if est <= room or (i == 0 and not running):   # (never none running)
            start.append(i)
            room -= est
        elif i == 0 and oldest_wait >= patience:
            break                                      # the oldest has waited long enough: none passes it
    return start


class JobPort(ctypes.Structure):
    _fields_ = [('CompletionKey', ctypes.c_void_p), ('CompletionPort', wt.HANDLE)]


class ProcMemory(ctypes.Structure):
    _fields_ = [('cb', wt.DWORD), ('PageFaultCount', wt.DWORD)] + \
               [(n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                                              'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                                              'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]


class MemStatusEx(ctypes.Structure):
    _fields_ = [('dwLength', wt.DWORD), ('dwMemoryLoad', wt.DWORD)] + \
               [(n, ctypes.c_ulonglong) for n in ('total', 'avail', 'tpf', 'apf', 'tv', 'av', 'aev')]


JOB_OBJECT_ASSOCIATE_COMPLETION_PORT_INFORMATION = 7
JOB_OBJECT_MSG_NEW_PROCESS, JOB_OBJECT_MSG_EXIT_PROCESS, JOB_OBJECT_MSG_ABNORMAL_EXIT_PROCESS = 6, 7, 8
PROCESS_SUSPEND_RESUME, PROCESS_QUERY_LIMITED_INFORMATION = 0x0800, 0x1000
PROCESS_COMMAND_LINE_INFORMATION = 60   # NtQueryInformationProcess: the command line (Windows 8.1 and later)


class MemoryGate:
    """The gate on a job object (run_measured): the job's completion port names every process it gains and loses.
    estimates: {source file: estimated peak memory}; a compile of another source counts LIGHT_MEM."""

    def __init__(self, job, estimates):
        import threading
        k32 = self.k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        nt = self.nt = ctypes.WinDLL('ntdll')
        k32.CreateIoCompletionPort.restype = wt.HANDLE
        k32.CreateIoCompletionPort.argtypes = [wt.HANDLE, wt.HANDLE, ctypes.c_size_t, wt.DWORD]
        k32.GetQueuedCompletionStatus.argtypes = [wt.HANDLE, ctypes.POINTER(wt.DWORD), ctypes.POINTER(ctypes.c_size_t),
                                                  ctypes.POINTER(ctypes.c_void_p), wt.DWORD]
        k32.OpenProcess.restype = wt.HANDLE
        k32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        k32.CloseHandle.argtypes = [wt.HANDLE]
        k32.K32GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD]
        k32.GlobalMemoryStatusEx.argtypes = [ctypes.c_void_p]
        k32.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
        nt.NtSuspendProcess.argtypes = nt.NtResumeProcess.argtypes = [wt.HANDLE]
        nt.NtQueryInformationProcess.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.ULONG,
                                                 ctypes.POINTER(wt.ULONG)]
        self.est = {Path(f).name.lower(): e for f, e in estimates.items()}
        self.waiting = []      # [pid, handle, estimate, since], oldest first
        self.running = {}      # pid: (handle, estimate)
        self.compiles = self.most = self.held = 0
        self.held_s, self.least_free = 0.0, None
        self.stop = threading.Event()
        self.port = k32.CreateIoCompletionPort(wt.HANDLE(-1), None, 0, 1)
        link = JobPort(None, self.port)
        self.ok = bool(self.port) and bool(k32.SetInformationJobObject(
            job, JOB_OBJECT_ASSOCIATE_COMPLETION_PORT_INFORMATION, ctypes.byref(link), ctypes.sizeof(link))) and \
            self.command_line(-1) is not None   # (a Windows or Wine without the query: no gate)
        self.thread = threading.Thread(target=self.run, name='memory gate', daemon=True)

    def command_line(self, h):
        buf, got = ctypes.create_string_buffer(1 << 16), wt.ULONG()
        if self.nt.NtQueryInformationProcess(wt.HANDLE(h), PROCESS_COMMAND_LINE_INFORMATION, buf, len(buf),
                                             ctypes.byref(got)) != 0:
            return None
        n = int.from_bytes(buf.raw[:2], 'little')             # UNICODE_STRING: Length, MaximumLength, pad, Buffer
        at = int.from_bytes(buf.raw[8:16], 'little')
        return ctypes.wstring_at(at, n // 2) if at else ''

    def parent(self, h):
        info = (ctypes.c_size_t * 6)()                         # PROCESS_BASIC_INFORMATION: the parent's id is last
        ok = self.nt.NtQueryInformationProcess(h, 0, info, ctypes.sizeof(info), None) == 0
        return info[5] if ok else 0

    def commit(self, h):
        m = ProcMemory()
        m.cb = ctypes.sizeof(m)
        return m.PagefileUsage if self.k32.K32GetProcessMemoryInfo(h, ctypes.byref(m), m.cb) else 0

    def free(self):
        s = MemStatusEx()
        s.dwLength = ctypes.sizeof(s)
        self.k32.GlobalMemoryStatusEx(ctypes.byref(s))
        return min(s.avail, s.apf)

    def new(self, pid):
        h = self.k32.OpenProcess(PROCESS_SUSPEND_RESUME | PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return                                             # gone already
        m = GATE_SOURCE.search(self.command_line(h) or '')
        # not Ninja itself, the linker or the resource compiler; nor a process a compile started (a wrapper such as a
        # compiler cache or a launcher runs the real compiler as its child with the same arguments), which is that
        # compile, counted for it already
        if not m or self.parent(h) in self.running:
            self.k32.CloseHandle(h)
            return
        self.nt.NtSuspendProcess(h)
        self.waiting.append([pid, h, self.est.get(Path(m.group(1) or m.group(2)).name.lower(), LIGHT_MEM), time.monotonic()])
        self.compiles += 1

    def gone(self, pid):
        if pid in self.running:
            self.k32.CloseHandle(self.running.pop(pid)[0])
        for w in [w for w in self.waiting if w[0] == pid]:   # (killed while held)
            self.waiting.remove(w)
            self.k32.CloseHandle(w[1])

    def tick(self):
        if not self.waiting:
            return
        free, now = self.free(), time.monotonic()
        growth = sum(max(0, est - self.commit(h)) for h, est in self.running.values())
        pick = gate_pick([w[2] for w in self.waiting], growth, free, len(self.running), now - self.waiting[0][3])
        for i in pick:
            pid, h, est, since = self.waiting[i]
            self.nt.NtResumeProcess(h)
            self.running[pid] = (h, est)
            self.held_s += now - since
            self.held += now - since >= GATE_TICK
        self.waiting = [w for i, w in enumerate(self.waiting) if i not in pick]
        self.most = max(self.most, len(self.running))

    def say(self):
        free = self.free()
        self.least_free = free if self.least_free is None else min(self.least_free, free)
        print(f'build_game: memory gate: {len(self.running)} compiling, {len(self.waiting)} held back, '
              f'{free / GiB:.1f} GiB free', flush=True)

    def run(self):
        n, key, ov, said = wt.DWORD(), ctypes.c_size_t(), ctypes.c_void_p(), time.monotonic()
        try:
            while not self.stop.is_set():
                if self.k32.GetQueuedCompletionStatus(self.port, ctypes.byref(n), ctypes.byref(key), ctypes.byref(ov),
                                                      int(GATE_TICK * 1000)):
                    if n.value == JOB_OBJECT_MSG_NEW_PROCESS:
                        self.new(ov.value or 0)
                    elif n.value in (JOB_OBJECT_MSG_EXIT_PROCESS, JOB_OBJECT_MSG_ABNORMAL_EXIT_PROCESS):
                        self.gone(ov.value or 0)
                self.tick()
                if time.monotonic() - said >= GATE_SAY and (self.running or self.waiting):
                    said = time.monotonic()
                    self.say()
        finally:
            for w in self.waiting:                             # (never left suspended)
                self.nt.NtResumeProcess(w[1])

    def summary(self):
        self.stop.set()
        self.thread.join(5)
        least = f', the least free {self.least_free / GiB:.1f} GiB' if self.least_free is not None else ''
        return (f'memory gate: {self.compiles} compiles, at most {self.most} at once, {self.held} held back for '
                f'{self.held_s:.0f} s in all{least}')


def run_measured(cmd, env, gate=None):
    """Run cmd in its own job object. Returns (exit code, seconds, peak bytes of one process, peak bytes of all).
    Off Windows: the largest resident set of one child process, and 0 for all of them (no job accounting).
    gate: {source file: estimated peak} to start each compile of the job when memory holds it (MemoryGate)."""
    if not WIN:
        t0 = time.monotonic()
        code = subprocess.call(cmd, cwd=ROOT, env=env, preexec_fn=lambda: os.nice(10))   # below normal, as on Windows
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024   # KiB on Linux
        return code, time.monotonic() - t0, peak, 0
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.CreateJobObjectW.restype = wt.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
    k32.QueryInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD, ctypes.c_void_p]
    k32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
    k32.CloseHandle.argtypes = [wt.HANDLE]
    ntdll = ctypes.WinDLL('ntdll')
    ntdll.NtResumeProcess.argtypes = [wt.HANDLE]

    job = k32.CreateJobObjectW(None, None)
    info = ExtendedLimits()
    # Ctrl+C stops every compile; every process of the job runs below normal priority, so the PC stays usable while
    # the compiles take every core
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_PRIORITY_CLASS
    info.BasicLimitInformation.PriorityClass = BELOW_NORMAL_PRIORITY_CLASS
    k32.SetInformationJobObject(job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info))
    g = MemoryGate(job, gate) if gate else None
    if g and not g.ok:
        print('build_game: warning: the memory gate did not start; the compiles run as Ninja starts them', file=sys.stderr)
        g = None
    if g:
        g.thread.start()
    t0 = time.monotonic()
    # Suspended until it is in the job, so the compilers it starts are counted from the first one.
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, creationflags=CREATE_SUSPENDED | BELOW_NORMAL_PRIORITY_CLASS)
    if not k32.AssignProcessToJobObject(job, int(proc._handle)):
        print(f'build_game: cannot track memory (error {ctypes.get_last_error()})', file=sys.stderr)
    ntdll.NtResumeProcess(int(proc._handle))
    try:
        code = proc.wait()
    except KeyboardInterrupt:
        k32.CloseHandle(job)
        raise
    dt = time.monotonic() - t0
    if g:
        print(f'build_game: {g.summary()}', flush=True)
    k32.QueryInformationJobObject(job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info),
                                  ctypes.sizeof(info), None)
    k32.CloseHandle(job)
    return code, dt, info.PeakProcessMemoryUsed, info.PeakJobMemoryUsed


def total_ram():
    if not WIN:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')

    class MemStatus(ctypes.Structure):
        _fields_ = [('dwLength', wt.DWORD), ('dwMemoryLoad', wt.DWORD)] + \
                   [(n, ctypes.c_ulonglong) for n in ('total', 'avail', 'tpf', 'apf', 'tv', 'av', 'aev')]
    s = MemStatus()
    s.dwLength = ctypes.sizeof(s)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.total


def memory_status():
    """(total, available) physical memory in bytes."""
    if not WIN:
        info = {}
        try:
            for line in open('/proc/meminfo'):
                k, v = line.split(':', 1)
                info[k] = int(v.split()[0]) * 1024
        except OSError:
            pass
        total = info.get('MemTotal') or total_ram()
        return total, info.get('MemAvailable', total // 2)

    class MemStatus(ctypes.Structure):
        _fields_ = [('dwLength', wt.DWORD), ('dwMemoryLoad', wt.DWORD)] + \
                   [(n, ctypes.c_ulonglong) for n in ('total', 'avail', 'tpf', 'apf', 'tv', 'av', 'aev')]
    s = MemStatus()
    s.dwLength = ctypes.sizeof(s)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.total, s.avail


def compile_budget():
    """(bytes the compile jobs may use together, how it was decided). REKARINA_BUILD_MEM_GB=<n> sets it (a test, or a
    player who knows better); else the memory available now less MEM_RESERVE, at most MEM_BUDGET, at least 1 GiB."""
    forced = os.environ.get('REKARINA_BUILD_MEM_GB', '').strip()
    if forced:
        try:
            return max(GiB // 2, int(float(forced) * GiB)), f'REKARINA_BUILD_MEM_GB={forced}'
        except ValueError:
            pass
    total, avail = memory_status()
    budget = min(MEM_BUDGET, max(GiB, avail - MEM_RESERVE))
    return budget, f'{avail / GiB:.1f} of {total / GiB:.1f} GiB available, {MEM_RESERVE / GiB:.1f} GiB kept for the PC'


def chunk_memory(gen, mods):
    """{generated C file: its compile's estimated peak memory} from each module's manifest chunk_info (the bytes and the
    functions of each chunk): LIGHT_MEM, or for a giant function's chunk GIANT_MEM_BASE + GIANT_MEM_PER_MB per MB."""
    est = {}
    for mod, files in mods.items():
        info = {}
        try:
            data = json.loads((gen / mod / 'manifest.json').read_text())
            for c in data.get('chunk_info', []) if isinstance(data, dict) else []:
                info[c.get('file')] = (int(c.get('bytes', 0)), max(1, int(c.get('functions', 1))))
        except (OSError, ValueError, TypeError, AttributeError):
            info = {}
        for f in files:
            size, funcs = info.get(f.name, (0, 1))   # not a chunk (<mod>_table.c, data tables: 480 MiB at most)
            per_fn = size / funcs
            est[f] = int(GIANT_MEM_BASE + GIANT_MEM_PER_MB * per_fn / 1e6) if per_fn >= GIANT_FN_BYTES else LIGHT_MEM
    return est


def gate_available():
    """Whether the memory gate (MemoryGate) works here: Windows (or Wine) with job completion ports and the command line
    query. Off Windows, with REKARINA_BUILD_MEM_GB or with REKARINA_BUILD_GATE=0 the plan from the memory available at
    the start stands alone."""
    if not WIN or os.environ.get('REKARINA_BUILD_MEM_GB', '').strip() or os.environ.get('REKARINA_BUILD_GATE') == '0':
        return False
    try:
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.CreateJobObjectW.restype = wt.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]
        k32.CloseHandle.argtypes = [wt.HANDLE]
        job = k32.CreateJobObjectW(None, None)
        g = MemoryGate(job, {})
        ok = g.ok
        k32.CloseHandle(g.port)
        k32.CloseHandle(job)
        return ok
    except (OSError, AttributeError, ValueError):
        return False


def plan_jobs(estimates, cpus, budget):
    """(jobs, giants at once) for compiles of these estimated peaks within budget bytes: at most half the budget goes
    to the giant chunks (at least one of them runs, whatever the budget), the rest to light compiles of LIGHT_MEM, so
    giants * the largest giant + (jobs - giants) * LIGHT_MEM stays within the budget; jobs at most cpus."""
    cpus = max(1, cpus)
    giant = max((e for e in estimates if e > LIGHT_MEM), default=0)
    if not giant:
        return max(1, min(cpus, budget // LIGHT_MEM)), 0
    giants = max(1, min(cpus, int(budget // 2 // giant)))
    jobs = giants + max(0, int((budget - giants * giant) // LIGHT_MEM))
    jobs = max(1, min(cpus, jobs))
    return jobs, min(giants, jobs)


def pe_sections(path):
    """{section name: virtual size} and SizeOfImage of a PE32+ file."""
    data = path.read_bytes()[:4096]
    pe = int.from_bytes(data[0x3C:0x40], 'little')
    nsec = int.from_bytes(data[pe + 6:pe + 8], 'little')
    opt = pe + 24
    image = int.from_bytes(data[opt + 56:opt + 60], 'little')
    table = opt + int.from_bytes(data[pe + 20:pe + 22], 'little')
    out = {}
    for i in range(nsec):
        s = data[table + 40 * i:table + 40 * (i + 1)]
        out[s[:8].rstrip(b'\0').decode(errors='replace')] = int.from_bytes(s[8:12], 'little')
    return out, image


def elf_sections(path):
    """{section name: size} of the allocated sections of an ELF64 file, and the extent they span in memory."""
    with open(path, 'rb') as f:
        head = f.read(64)
        shoff = int.from_bytes(head[0x28:0x30], 'little')
        shentsize, shnum, shstrndx = (int.from_bytes(head[o:o + 2], 'little') for o in (0x3A, 0x3C, 0x3E))
        f.seek(shoff)
        table = f.read(shentsize * shnum)
        sh = [table[i * shentsize:(i + 1) * shentsize] for i in range(shnum)]
        f.seek(int.from_bytes(sh[shstrndx][0x18:0x20], 'little'))
        names = f.read(int.from_bytes(sh[shstrndx][0x20:0x28], 'little'))
    out, lo, hi = {}, None, 0
    for s in sh:
        flags = int.from_bytes(s[0x08:0x10], 'little')
        addr, size = int.from_bytes(s[0x10:0x18], 'little'), int.from_bytes(s[0x20:0x28], 'little')
        if not flags & 2 or not size:   # SHF_ALLOC
            continue
        name = names[int.from_bytes(s[0:4], 'little'):].split(b'\0', 1)[0].decode(errors='replace')
        out[name] = out.get(name, 0) + size
        lo = addr if lo is None else min(lo, addr)
        hi = max(hi, addr + size)
    return out, hi - (lo or 0)


def objdump_tool():
    """The disassembler for tools/check_isa.py: llvm-objdump of --toolchain, else MSYS2's or the one on PATH."""
    suffix = '.exe' if WIN else ''
    for d in (TOOLCHAIN_BIN, UCRT if WIN else None):
        for name in ('llvm-objdump', 'objdump'):
            if d and (d / (name + suffix)).is_file():
                return str(d / (name + suffix))
    return check_isa.find_objdump()


def mib(n):
    return f'{n / (1 << 20):,.1f} MiB'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--version', default='v121', help='game version (default v121)')
    ap.add_argument('--cpu', choices=('v2', 'v3'), default='v3',
                    help='target CPU level: v3 (AVX2/FMA, default) or v2 (SSE4.2, for older CPUs; slower)')
    ap.add_argument('--gen', type=Path, help='generated C directory (default generated/<version>)')
    ap.add_argument('--out', type=Path, help='build directory (default build/<version>)')
    ap.add_argument('--modules', help='comma-separated subset of the layout modules (default all)')
    ap.add_argument('--jobs', type=int, help='parallel compiles (default: by memory budget and CPU count)')
    ap.add_argument('--compile-only', action='store_true', help='compile, do not link')
    ap.add_argument('--runtime', type=Path, help='runtime source directory (default runtime)')
    ap.add_argument('--full-build', action='store_true',
                    help="compile the generated C even where build/<version>'s objects could be reused")
    ap.add_argument('--skip-runtime', default='',
                    help='comma-separated runtime/*.c names to leave out (the link needs another definition of what they '
                         'define)')
    ap.add_argument('--exefs', type=Path,
                    help="rekarina.exe's default ExeFS directory (default <work dir>/extracted/<version>/exefs, the work "
                         "dir being the [work] dir of config.local.toml, else build)")
    ap.add_argument('--romfs', type=Path, help="rekarina.exe's default RomFS directory with romfs.bin and index.json "
                                               '(default <work dir>/extracted/<version>/romfs)')
    ap.add_argument('--no-romfs', action='store_true',
                    help='build even when the RomFS directory has no index.json (the exe then stops at start)')
    ap.add_argument('--profile', type=Path, help="rekarina.exe's default profile directory (default build/profile for v121, "
                                                 "build/profile-<version> otherwise, so trying another version leaves "
                                                 "the main profile alone; saves load on 1.2.1 and 1.4.3 alike)")
    ap.add_argument('--toolchain', type=Path, help='llvm-mingw directory (with bin/clang.exe) to build with instead of '
                                                   'MSYS2 UCRT64 clang; on Linux an LLVM directory (bin/clang, '
                                                   'bin/llvm-ar) instead of the clang on PATH')
    ap.add_argument('--ninja', type=Path, help='ninja.exe to build with (default: MSYS2 UCRT64, else bin/ninja.exe of the '
                                               'release folder)')
    ap.add_argument('--sysroot', type=Path, help='Linux: the glibc headers and libraries to compile and link against '
                                                 'instead of the host\'s (the release\'s llvm/sysroot)')
    ap.add_argument('--deps', type=Path, help=f'with --toolchain: include/, lib/ and bin/ of Vulkan, SDL3, Opus, libvpx '
                                              f'and VMA (default {DEFAULT_DEPS.relative_to(ROOT).as_posix()})')
    ap.add_argument('--make-runtime-lib', action='store_true',
                    help=f'build only <out>/{LIB_NAME} from runtime/ (no generated C, no rekarina.exe)')
    ap.add_argument('--runtime-lib', type=Path, help=f'link against this {LIB_NAME} instead of compiling runtime/ '
                                                    f'({", ".join(sorted(INSTALL_SIDE))} and modules.c still compile)')
    ap.add_argument('--pgo', metavar='gen|FILE',
                    help='profile-guided optimisation, generated code at -O2: gen builds an instrumented rekarina.exe '
                         '(needs --toolchain on Windows: MSYS2 clang ships no profile runtime); FILE builds with that '
                         '.profdata (tools/pgo.py merge)')
    ap.add_argument('--expected', type=Path,
                    help='expected hashes (default tools/data/expected-<display version>.json)')
    args = ap.parse_args()
    set_cpu(args.cpu)
    set_lto(args.cpu)

    gen = (args.gen or ROOT / 'generated' / args.version).resolve()
    out = (args.out or ROOT / 'build' / args.version).resolve()
    if args.pgo:
        if args.pgo == 'gen' and WIN and not args.toolchain:
            sys.exit('--pgo gen: MSYS2 UCRT64 clang ships no profile runtime (clang_rt.profile); pass --toolchain with '
                     'an llvm-mingw directory')
        set_pgo(args.pgo, out)
    extracted = None if args.exefs and args.romfs else work_dir() / 'extracted' / args.version
    paths = {'exefs': (args.exefs or extracted / 'exefs').resolve(),
             'romfs': (args.romfs or extracted / 'romfs').resolve(),
             'profile': (args.profile or ROOT / 'build' / ('profile' if args.version == 'v121'
                                                           else f'profile-{args.version}')).resolve()}
    has_romfs = (paths['romfs'] / 'index.json').is_file()
    if not (has_romfs or args.no_romfs or args.make_runtime_lib or args.compile_only):
        sys.exit(f'{paths["romfs"] / "index.json"} not found: without the RomFS the game stops at start. Run '
                 'tools/extract.py (it writes extracted/<version> into the [work] dir of config.local.toml), pass '
                 '--romfs DIR, or --no-romfs to build without it')
    expected = args.expected
    if not expected and args.version in OTHER_GAMES:
        expected = ROOT / 'tools' / 'data' / f'{OTHER_GAMES[args.version][0]}.json'
        RT_FLAGS.extend(OTHER_GAMES[args.version][2])
        GEN_FLAGS.extend(OTHER_GAMES[args.version][2])
    if not expected:
        if args.version not in DISPLAY_VERSIONS:
            sys.exit(f'no expected hashes known for {args.version}; pass --expected')
        expected = ROOT / 'tools' / 'data' / f'expected-v{DISPLAY_VERSIONS[args.version]}.json'
    hashes = json.loads(Path(expected).read_text())['exefs_sha256']
    global CLANG, AR, WINDRES, NINJA, TOOLCHAIN_BIN, DEPS
    if args.ninja:
        NINJA = args.ninja.resolve()
    elif not NINJA.is_file() and (ROOT / 'bin' / ('ninja' + ('.exe' if WIN else ''))).is_file():
        NINJA = ROOT / 'bin' / ('ninja' + ('.exe' if WIN else ''))   # the release folder (tools/make_release.py)
    dlls = SDL3_DLLS
    suffix = '.exe' if WIN else ''
    if args.toolchain:
        TOOLCHAIN_BIN = (args.toolchain / 'bin').resolve()
        CLANG = TOOLCHAIN_BIN / f'clang{suffix}'
        AR = TOOLCHAIN_BIN / f'llvm-ar{suffix}'
        WINDRES = TOOLCHAIN_BIN / 'llvm-windres.exe'
    if args.toolchain and WIN or args.deps and (args.toolchain or not WIN):   # Linux: --deps only, never deps-mingw
        DEPS = (args.deps or DEFAULT_DEPS).resolve()
        dlls = sorted((DEPS / 'bin').glob('*.dll')) if WIN else []
    if DEPS and not WIN:
        # A release (--deps): Opus, libvpx and zstd go into the executable, so it needs no more than libc, Vulkan and
        # the libSDL3.so.0 beside it
        for lib in ('opus', 'vpx', 'zstd'):
            LINK_LIBS[LINK_LIBS.index(f'-l{lib}')] = f'-l:lib{lib}.a'
    if args.version == 'oot' and not (WIN and DEPS):   # Windows release deps already link dav1d.
        LINK_LIBS.insert(LINK_LIBS.index('-l:libstdc++.a'), '-l:libdav1d.a' if WIN or DEPS else '-ldav1d')
    if args.sysroot and not WIN:
        # Every compile and the link: the game then needs no newer glibc than the sysroot's (the Steam Runtime's)
        sysroot = ['--target=x86_64-linux-gnu', f'--sysroot={cmd_path(args.sysroot.resolve())}']
        GEN_FLAGS[:0] = sysroot
        RT_FLAGS[:0] = sysroot
        LINK_FLAGS[:0] = sysroot
    for tool in (CLANG, NINJA):
        if not tool.is_file():
            sys.exit(f'{tool} not found: install ' + ('MSYS2 UCRT64 clang, lld and ninja' if WIN else 'clang, lld and ninja')
                     + ' (docs/building.md)')
    if not (gen / 'layout.json').is_file():
        sys.exit(f'{gen / "layout.json"} not found: run the generator first')

    layout = read_layout(gen)
    names = [n for n, _, _ in layout]
    wanted = args.modules.split(',') if args.modules else names
    unknown = [m for m in wanted if m not in names]
    if unknown:
        sys.exit(f'unknown modules {unknown}; the layout has {names}')
    linked = [m for m in names if m in wanted]
    # --make-runtime-lib builds runtime/ only, so it never reads the generated chunks
    mods = {} if args.make_runtime_lib else {m: chunk_files(gen, m) for m in linked}

    cpus = os.cpu_count() or 1
    gated = gate_available()
    if gated:   # the plan is the most the PC's memory could hold; the gate starts each compile when it fits now
        total, avail = memory_status()
        budget = min(MEM_BUDGET, max(GiB, total - MEM_RESERVE))
        why_budget = (f'{avail / GiB:.1f} of {total / GiB:.1f} GiB available now; each compile starts when the memory '
                      f'free then holds it')
    else:
        budget, why_budget = compile_budget()
    estimates = chunk_memory(gen, mods)
    giants = {f: e for f, e in estimates.items() if e > LIGHT_MEM}
    jobs, giant_depth = plan_jobs(list(estimates.values()), cpus, budget)
    if args.jobs:   # given: that many jobs; the giants still at most as the budget allows
        jobs, giant_depth = args.jobs, max(1, min(args.jobs, giant_depth))
    print(f'build_game: compile memory {budget / GiB:.1f} GiB ({why_budget}): {jobs} jobs, at most {giant_depth} of the '
          f'{len(giants)} giant chunks at once (the largest about {max(giants.values(), default=0) / GiB:.1f} GiB)',
          flush=True)
    missing = [n for n in names if n not in hashes]
    if missing:
        sys.exit(f'{expected}: no exefs_sha256 for {missing}')
    write_if_changed(out / 'modules.c', module_glue(layout, linked, hashes, paths, args.version))
    if has_romfs:
        count, size, dt = romfs_index.write(paths['romfs'], out)
        print(f'romfs index: {count} files, {mib(size)}, {dt:.1f} s')
    else:
        romfs_index.write_placeholder(out)
        print(f'build_game: warning: no {paths["romfs"] / "index.json"}; {EXE} will have no RomFS', file=sys.stderr)
    copy_romfs_patches(args.version, out)
    global RUNTIME
    RUNTIME = (args.runtime or RUNTIME).resolve()
    stale = stale_shaders(RUNTIME)
    if stale:
        sys.exit(f'{stale[0]}: its SPIR-V header no longer matches; run python tools/build_shaders.py')
    skip = {s for s in args.skip_runtime.split(',') if s}
    main_out = (ROOT / 'build' / args.version).resolve()
    pairs = None
    if args.make_runtime_lib:
        mods, linked, why = {}, [], ''
    elif args.full_build:
        why = '--full-build'
    elif args.toolchain:
        why = f'--toolchain {flag_path(args.toolchain)}'
    elif out == main_out:
        why = ''
    else:
        pairs, why = main_objects(gen, main_out, RUNTIME, mods)
    if pairs is not None:
        link_objects(out, pairs)
        print(f'build_game: reusing {len(pairs)} generated objects of {flag_path(main_out)} (hardlinks)')
    else:
        if out != main_out:   # forks hold hardlinks to the main build's objects; the main build keeps its own
            unlink_shared(out, mods)
        if why:
            print(f'build_game: compiling all generated C: {why}')
    runtime_lib = args.runtime_lib.resolve() if args.runtime_lib else None
    if runtime_lib and not runtime_lib.is_file():
        sys.exit(f'{runtime_lib} not found: build it with --make-runtime-lib')
    # The library leaves INSTALL_SIDE out, so a release that ships without one of them links with undefined symbols.
    missing = sorted(n for n in INSTALL_SIDE - skip if not (RUNTIME / n).is_file())
    if runtime_lib and missing:
        sys.exit(f'build_game: {", ".join(missing)} missing from {flag_path(RUNTIME)}: the release must ship every '
                 f'INSTALL_SIDE source beside {runtime_lib.name}')
    edges = {}
    text, objs, exe = ninja_file(gen, out, mods, jobs, skip, pairs is not None, runtime_lib,
                                 args.make_runtime_lib, edges, giants, giant_depth)
    write_if_changed(out / 'build.ninja', text)

    env = dict(os.environ)
    if WIN or TOOLCHAIN_BIN:
        env['PATH'] = str(TOOLCHAIN_BIN or UCRT) + os.pathsep + os.environ.get('PATH', '')
    base = [str(NINJA), '-f', str(out / 'build.ninja'), '-j', str(jobs)]
    nfiles = sum(len(v) for v in mods.values())
    print(f'build_game: {nfiles} generated files in {len(linked)} modules ({", ".join(linked)}), {jobs} jobs', flush=True)

    drop_stale_by_content(out, edges, env)   # file times of release sources do not say what changed
    code, dt, peak_one, peak_all = run_measured(base + ['objects'], env, estimates if gated else None)
    print(f'compile: {dt:.1f} s, peak {mib(peak_one)} per process' + (f', {mib(peak_all)} all processes' if WIN else ''))
    if code:
        sys.exit(code)
    for mod, files in objs.items():
        size = sum(o.stat().st_size for o in files)
        print(f'  {mod:8} {len(files):5} objects {mib(size):>14}')
    verify_objects(out, objs, pairs is not None, main_out, pairs, base, env)
    record_inputs(out, edges, env, [o for o in edges if o != exe])
    if args.cpu == 'v2':
        # Inline assembly, target attributes and intrinsics pass -march=x86-64-v2 by; one vmovdqa of a YMM register
        # stops a CPU without AVX (tools/check_isa.py)
        fail = check_isa.check_objects(objs['runtime'], objdump_tool(), out / 'obj' / 'runtime' / 'isa-v2.json')
        if fail:
            sys.exit(f'build_game: {fail}')
    if args.compile_only:
        return
    if args.make_runtime_lib:
        exe.unlink(missing_ok=True)   # ar rcs would keep members of deleted sources
    code, dt, peak_one, _ = run_measured(base + [flag_path(exe)], env)
    print(f'link: {dt:.1f} s, peak {mib(peak_one)}')
    if code:
        print(f'build_game: link failed, {flag_path(exe)} is the old one (a running rekarina.exe from this dir locks it)',
              file=sys.stderr)
        sys.exit(code)
    if exe in edges:
        record_inputs(out, edges, env, [exe])
    if args.make_runtime_lib:
        print(f'{below_root(exe)}: {mib(exe.stat().st_size)}')
        return
    for dll in dlls:   # dynamic SDL3 (zlib) and its libiconv (LGPL, unmodified DLL), found next to rekarina.exe
        shutil.copy2(dll, exe.parent / dll.name)
    sections, image = pe_sections(exe) if WIN else elf_sections(exe)
    print(f'{below_root(exe)}: file {mib(exe.stat().st_size)}, '
          f'image {mib(image)}; ' + ', '.join(f'{k} {mib(v)}' for k, v in sections.items() if k.startswith('.') and v))
    if image > 3 * GiB // 2:
        print('build_game: warning: image over 1.5 GiB; RIP-relative code reaches only 2 GiB', file=sys.stderr)


if __name__ == '__main__':
    main()
