"""Package the OoT (Switch 2) build into a folder to play from: Rekarina.exe (Rekarina on Linux) and what it reads beside
itself.

The folder holds the exe and its DLLs, the shader translator, the prepared shader and pipeline files, the ExeFS the
exe loads (the uncompressed one, tools/nso_plain.py), the Nintendo extension shared fonts of each prompt style
(tools/make_ext_font.py), the RomFS overlays (tools/make_oot_prompts.py: stand-ins for the dump's damaged textures, Xbox prompts) and a rekarina.ini that names the player's
.nup as the RomFS (runtime/main.c portable_paths: REKARINA_ROMFS) with prompts = auto and the frame limit at the monitor's
rate (runtime/frame_rate.c, at most 240). The RomFS stays in the .nup. Saves go to <folder>/profile. An existing
rekarina.ini is kept (the player's settings; a totk.ini of an earlier package becomes it), as are profile and the shader cache's new entries. A new rekarina.ini also
gets the --set lines: the setup's first graphics settings for the machine's graphics card (installer/components.py
preset).

On Linux the game and the translator have no suffix, and --lib names the libraries the game finds beside itself (the
release's libSDL3.so.0: tools/build_game.py links against the Steam Runtime sysroot with an $ORIGIN rpath).

The folder contains game code and data derived from the player's game: for their own machines only.

    .venv/Scripts/python tools/package_oot.py --nup <container> [--build build/oot] [--exefs ../build/oot/exefs-plain]
                                              [--out ../OoT] [--prompts auto] [--lang ru]
                                              [--set key=value ...] [--preset-note TEXT]
"""
import argparse
import gzip
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))   # tools/pack_shader_cache.py
WIN = os.name == 'nt'
EXE = '.exe' if WIN else ''
# nvngx_dlss.dll and AMD's split FSR SDK 2 libraries are loaded at run time when they are beside the exe
# ... and Streamline's for DLSS Frame Generation, with our library between the game and Streamline (runtime/dlssg, built
# against the Streamline SDK): totk_dlssg.dll in the build folder, rekarina_dlssg.dll in the package (RENAMED).
# NVIDIA's and AMD's are copied when the build folder has them (a developer's own copies, the setup's --dlls); the
# setup itself downloads them straight into the game folder (installer/components.py), no release carries them.
DLLS = ('SDL3.dll', 'libopus-0.dll', 'libiconv-2.dll', 'nvngx_dlss.dll', 'totk_dlssg.dll',
        'rekarina_dlssg.dll', 'renhdrum.dll',
        'amd_fidelityfx_loader_vk.dll', 'amd_fidelityfx_upscaler_vk.dll', 'amd_fidelityfx_framegeneration_vk.dll',
        'sl.interposer.dll', 'sl.common.dll', 'sl.dlss.dll', 'sl.dlss_g.dll', 'sl.reflex.dll', 'sl.pcl.dll', 'nvngx_dlssg.dll', 'NvLowLatencyVk.dll')
SEED_KEYS = ROOT / 'tools' / 'data' / 'pipeline-keys-oot.bin.gz'   # tools/make_seed_keys.py writes it
SEED_MESH_KEYS = ROOT / 'tools' / 'data' / 'pipeline-keys-mesh-oot.bin.gz'   # and the mesh pipelines' (vk_pipeline.c mk_open)
SEED_COMPUTE = ROOT / 'tools' / 'data' / 'compute-used-oot.bin.gz'   # and the compute shaders dispatched (vk_pipeline.c cu_load)
RENAMED = {'totk_dlssg.dll': 'rekarina_dlssg.dll'}   # build name -> the name in the package
OLD_NAMES = ('totk_xlate.exe', 'totk_xlate.id', 'totk_dlssg.dll')
STATE = ('pipeline-cache.bin', 'pipeline-keys.bin', 'pipeline-keys-mesh.bin', 'shader-answers.bin', 'shader-programs.bin',
         'shader-prepared.bin', 'shader-prepare-skip.txt')   # the last: variants the driver does not finish building (runtime/vk_pipeline.c)
# Every overlay folder tools/make_oot_prompts.py writes, and the files beside them that runtime/romfs.c reads: the sets'
# cmap tables for changing prompts in the running game (prompts-live.bin), the settings menu's previews
# (prompt-previews.bin) and the keyboard set's bound keys (prompts-keys.bin). Until 2026-11-15 the keyboard set's
# folder and those three files stayed in the build folder: a setup's game showed the Switch prompts for the keyboard and
# could not change sets while it ran (a player's report that not every button switched).
FOLDERS = ('romfs-patches', 'romfs-patches-xbox', 'romfs-patches-xbox-label', 'romfs-patches-playstation',
           'romfs-patches-playstation-label', 'romfs-patches-rekarina', 'romfs-patches-rekarina-label',
           'romfs-patches-pc', 'romfs-patches-fasttext',
           'shared-fonts-switch', 'shared-fonts-xbox',
           'shared-fonts-xbox-label')
PROMPT_FILES = ('prompts-live.bin', 'prompt-previews.bin', 'prompts-keys.bin', 'prompts-hint.bin',
                'prompts-ocarina.bin')   # the action hint's and the song screen's button images of every set
# Overlay folders that are there only when the game's files were the ones their tool knows: the action hint's layout for a
# pad mapped by label (make_oot_prompts.py hint_by_label)
OPTIONAL_FOLDERS = ('romfs-patches-hint-label',)


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


SINGLE_TRAILER, SINGLE_TRAILER_SIZE = b'RKSFX001', 56   # installer/launcher/rekarina_single.c, installer/oot.py
SINGLE_ASIDE = 'Rekarina-download' + EXE


def set_single_aside(exe):
    """The downloaded one-file Rekarina.exe (the setup and its launcher in one file, with the game's name) where the
    game goes: renamed to SINGLE_ASIDE before the game is copied, so the game never lands on a running copy of it and
    the download is not lost. A player who saved the download into the game folder had the launcher in the game's
    place: Play opened the launcher again (2026-11-20)."""
    try:
        with open(exe, 'rb') as f:
            size = f.seek(0, os.SEEK_END)
            f.seek(max(0, size - SINGLE_TRAILER_SIZE))
            tail = f.read(SINGLE_TRAILER_SIZE)
    except OSError:
        return
    if size <= SINGLE_TRAILER_SIZE or not tail.startswith(SINGLE_TRAILER) or \
            int.from_bytes(tail[8:16], 'little') + int.from_bytes(tail[16:24], 'little') + SINGLE_TRAILER_SIZE != size:
        return
    aside = exe.with_name(SINGLE_ASIDE)
    if aside.exists():
        aside.unlink()
    os.replace(exe, aside)   # a running exe can be renamed, not written over
    print(f'package_oot: {exe.name} was the downloaded launcher, not the game: moved to {aside.name}')


def same_file(src, dst):
    """dst is a copy of src as copy2 makes it: the same size and modification time (a second's slack for file systems
    that keep less). An update then copies nothing again of a shader pack of 2 GB that did not change."""
    try:
        a, b = os.stat(src), os.stat(dst)
    except OSError:
        return False
    return a.st_size == b.st_size and abs(a.st_mtime - b.st_mtime) < 1.0


def copy_tree(src, dst):
    n = 0
    if Path(src).resolve() == Path(dst).resolve():   # in place already: the installer extracts the ExeFS into the folder
        return sum(len(files) for _, _, files in os.walk(src))
    for dirpath, _, files in os.walk(src):
        rel = Path(dirpath).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            if not f.endswith('.tmp'):
                if not same_file(Path(dirpath) / f, dst / rel / f):
                    shutil.copy2(Path(dirpath) / f, dst / rel / f)
                n += 1
    return n


def newer(src, dst):
    """src should replace dst: dst is missing, or src was written later. The build folder's copies of the game's records
    (STATE) came from the game folder before the shader preparation (installer/oot.py seed_from_folder); what the game
    wrote into its own since (pipelines met in play, the cache it saved at its exit) must not be put back to that."""
    try:
        return not dst.exists() or os.stat(src).st_mtime > os.stat(dst).st_mtime + 1.0
    except OSError:
        return True


def keep_pack_entries(build_cache, folder_cache, folder_id, xlate_id):
    """Before the build's shader pack replaces the game folder's: when both were made under the same translator, the
    entries only the folder's pack holds go into the build's first (tools/pack_shader_cache.py merge), so a package never
    takes prepared shaders away. 1.2.3 copied a setup's pack over the folder's whatever each held, and a setup run from a
    fresh temp folder that had prepared the variants play met (340 MB) replaced a folder's 2 GB of every variant. Under
    another translator the folder's entries are of no use and the build's pack replaces it as before."""
    import pack_shader_cache as psc
    mine, theirs = build_cache / psc.PACK, folder_cache / psc.PACK
    if not (mine.is_file() and theirs.is_file() and folder_id and xlate_id.is_file() and folder_id == xlate_id.read_bytes()):
        return
    total, added = psc.merge(mine, theirs)
    if added:
        print(f'package_oot: {added} shader cache entries of the game folder kept ({total} in all)')


def graphics_text(settings, note):
    """The [graphics] section of a new rekarina.ini: the first settings the setup chose for the graphics card."""
    if not settings:
        return ''
    lines = ['', '[graphics]',
             '# First settings for this machine' + (f' ({note})' if note else '') + '. The setup wrote them once, with this',
             '# new file, and never changes them again: set them here or in the settings menu of the game.',
             '# dlaa = 1 switches the upscaler on (taa: dlss, fsr3 or fsr1, the spatial one; dlss: dlaa, quality, balanced, performance);',
             '# framegen: off, fsr3 or dlss (needs dlaa = 1); tex_quality: auto, full, half, quarter.']
    return '\n'.join(lines + [f'{k} = {v}' for k, v in settings]) + '\n'


def ini_text(nup, prompts, lang, settings=(), note=''):
    return f"""# rekarina.ini: the settings of Rekarina. The settings menu of the game (F1) writes this file too. A REKARINA_*
# environment variable named after a key overrides its line.

[paths]
# The game: the .nup container of the player's dump (read in place).
romfs = {nup.as_posix()}

[display]
# Borderless fullscreen on the window's monitor (F11 switches it while the game runs and remembers the choice).
fullscreen = 1

[frame rate]
# 30, 40, 45, 60, 90, 120, 144, any whole rate from 30 to 240, or display. Unset (the line commented): display, the
# monitor's rate, or a handheld's own limit (the Steam Deck's profile: 40, Windows handhelds: 60), which a line here
# would override.
# fps_limit = display
# Real-time frame step: game speed follows the clock down to 30 fps (steps of at most 1/30 s, as the console's own
# 30 fps mode); below 30 fps the game slows down.
realtime = 1

[performance]
# ASTC textures re-encoded to BC on the GPU: a quarter of the VRAM of RGBA8 (the house went from 11.3 GB, over
# the budget, to 8 GB, and from 17 to 48-50 fps after the camera turn).
astc_bc = 1

[input]
# Button prompts: auto (the controller or the keyboard you used last, changed while the game runs), nintendo (the
# game's own), xbox, playstation, rekarina (Xbox names on the game's own button shapes) or pc (the keyboard and mouse).
prompts = {prompts}

[language]
# The game's language as a Switch language code (ja, en-US, en-GB, fr, de, it, es, nl, pt, ru, ko, zh-Hans, ...); unset
# follows the Windows user locale. The game has Russian text; its voice acting has no Russian track.
lang = {lang}
""" + graphics_text(settings, note)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--nup', type=Path, required=True, help="the player's .nup")
    ap.add_argument('--build', type=Path, default=ROOT / 'build' / 'oot')
    ap.add_argument('--exefs', type=Path, default=ROOT.parent / 'build' / 'oot' / 'exefs-plain')
    ap.add_argument('--xlate', type=Path, default=ROOT / 'build' / 'xlate' / ('totk_xlate' + EXE))
    ap.add_argument('--lib', type=Path, action='append', default=[], help='a library to put beside the game (Linux: libSDL3.so.0)')
    ap.add_argument('--out', type=Path, default=ROOT.parent / 'OoT')
    ap.add_argument('--prompts', default='auto', choices=('auto', 'xbox', 'nintendo', 'playstation', 'rekarina'))
    ap.add_argument('--lang', default='ru', help='the game language for the new rekarina.ini (Switch code)')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='a line for the [graphics] section of a new rekarina.ini (a key of runtime/cfg.c, lower case)')
    ap.add_argument('--preset-note', default='', help='what the --set lines were chosen for, for the comment above them')
    a = ap.parse_args()
    settings = []
    for item in a.set:
        m = re.fullmatch(r'([a-z][a-z0-9_]*)=([A-Za-z0-9_.:-]+)', item)
        if not m:
            sys.exit(f'package_oot: --set {item}: not key=value')
        settings.append(m.groups())
    build, out = a.build.resolve(), a.out.resolve()
    need = [build / ('rekarina' + EXE), a.xlate, a.exefs / 'main', a.exefs / 'main.npdm', a.nup] + [build / d for d in FOLDERS] + \
        [build / f for f in PROMPT_FILES] + a.lib
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        sys.exit('package_oot: missing: ' + ', '.join(missing))
    if out == build:
        sys.exit('package_oot: --out must not be the build dir')

    out.mkdir(parents=True, exist_ok=True)
    # The translator the folder's shader pack was made under, before this package names its own: under the same one
    # the pack's entries stay valid and are kept (below)
    folder_id = next((f.read_bytes() for f in (out / 'rekarina_xlate.id', out / 'totk_xlate.id') if f.is_file()), None)
    set_single_aside(out / ('Rekarina' + EXE))
    copy(build / ('rekarina' + EXE), out / ('Rekarina' + EXE))
    for d in DLLS:
        if (build / d).exists():
            copy(build / d, out / RENAMED.get(d, d))
    for lib in a.lib:
        copy(lib, out / lib.name)
    copy(a.xlate, out / ('rekarina_xlate' + EXE))
    if a.xlate.with_suffix('.id').exists():   # the translator identity is this file's content: the shader cache stays valid
        copy(a.xlate.with_suffix('.id'), out / 'rekarina_xlate.id')
    for old in OLD_NAMES:   # what a package made before the files were renamed left in the folder
        if (out / old).is_file():
            (out / old).unlink()
    for f in STATE:
        if (build / f).exists() and newer(build / f, out / f):
            copy(build / f, out / f)
    for f in PROMPT_FILES:
        copy(build / f, out / f)
    # The pipelines the release's makers met in play (runtime/vk_pipeline.c keys_seed): hashes and render state only.
    # The game builds them at its start, so a first run does not build each at its first draw.
    if SEED_KEYS.is_file():
        (out / 'pipeline-keys-seed.bin').write_bytes(gzip.decompress(SEED_KEYS.read_bytes()))
    if SEED_MESH_KEYS.is_file():
        (out / 'pipeline-keys-mesh-seed.bin').write_bytes(gzip.decompress(SEED_MESH_KEYS.read_bytes()))
    if SEED_COMPUTE.is_file():
        (out / 'compute-used-seed.bin').write_bytes(gzip.decompress(SEED_COMPUTE.read_bytes()))
    n = 0
    for d in FOLDERS + tuple(d for d in OPTIONAL_FOLDERS if (build / d / 'manifest.tsv').is_file()):
        n += copy_tree(build / d, out / d)
    n += copy_tree(a.exefs, out / 'exefs')
    # The driver's cache of the shader preparation (runtime/vk_device.c), into a folder that has none: since 1.2.8 the
    # preparation only translates, and the game folder's own cache (what the game and the launcher's background work
    # built there) is newer than anything the build folder holds; its .bin and .toc go together, never one of them.
    if (build / 'nv-cache').is_dir() and not (out / 'nv-cache').exists():
        n += copy_tree(build / 'nv-cache', out / 'nv-cache')
    if (build / 'shader-cache').is_dir():
        keep_pack_entries(build / 'shader-cache', out / 'shader-cache', folder_id, a.xlate.with_suffix('.id'))
        n += copy_tree(build / 'shader-cache', out / 'shader-cache')
    (out / 'profile').mkdir(exist_ok=True)
    ini, old_ini = out / 'rekarina.ini', out / 'totk.ini'
    if not ini.exists() and old_ini.exists():   # the player's settings from before the file was renamed
        old_ini.replace(ini)
        print(f'package_oot: {old_ini.name} renamed to {ini.name}')
    if ini.exists():
        print(f'package_oot: kept {ini}')
    else:
        note = re.sub(r'[^\x20-\x7e]', '?', a.preset_note)   # one comment line, whatever the card calls itself
        ini.write_bytes(ini_text(a.nup.resolve(), a.prompts, a.lang, settings, note).encode('utf-8'))
    size = sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
    print(f'package_oot: {out}: Rekarina{EXE} and {n} files in folders, {size / 2**30:.2f} GiB')


if __name__ == '__main__':
    main()
