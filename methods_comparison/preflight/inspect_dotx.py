"""Inspiziert ECCM22-Vorlage und versucht den Reference-Stil zu erraten.

PASS: 'vancouver' oder 'numbered' im Template-Body gefunden, oder vorhandene
      Beispiel-Refs sehen wie [1], [2], ... aus.
WARN: kein klares Signal -> User-Decision (Vancouver/IEEE/author-year).
FAIL: Template-Datei nicht gefunden.
"""
import re
import sys
import zipfile
from pathlib import Path

DOTX = Path(
    os.environ.get("OUTPUT_ROOT", "./output")
    "/19_ECCM_Oslo_2026/ECCM22_AbstractID_Fullpaper_Lastname_Firstname.dotx"
)


def main() -> int:
    if not DOTX.exists():
        print(f"FAIL: template not found at {DOTX}")
        return 1

    try:
        with zipfile.ZipFile(DOTX) as z:
            body = z.read("word/document.xml").decode("utf-8", errors="ignore")
    except (zipfile.BadZipFile, KeyError) as e:
        print(f"FAIL: cannot parse {DOTX}: {e}")
        return 1

    ref_section = re.search(r"References.*?(?=</w:body>|$)", body, re.S)
    snippet = ref_section.group(0)[:2000] if ref_section else body[-3000:]

    if re.search(r"\[\d+\]", snippet) or "vancouver" in body.lower():
        style = "vancouver-numbered"
        print("PASS: numbered-Vancouver pattern detected")
        # Update timeline.md with reference style info
        timeline = Path("methods_comparison/timeline.md")
        existing = timeline.read_text() if timeline.exists() else ""
        if "## Reference style" in existing:
            existing = re.sub(
                r"## Reference style\n.*",
                f"## Reference style\n{style}\n",
                existing,
                flags=re.S,
            )
        else:
            existing += f"\n## Reference style\n{style}\n"
        timeline.write_text(existing)
        return 0

    if re.search(r"\(\w+, ?\d{4}\)", snippet):
        print("WARN: author-year pattern detected, NOT Vancouver. Plan Task 10.1 muss Stil tauschen.")
        return 0

    print("WARN: cannot detect reference style automatically. Open template manually.")
    print(f"  open '{DOTX}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
