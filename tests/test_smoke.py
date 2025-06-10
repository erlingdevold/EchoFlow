"""
Very small end-to-end check:

1.  Generates a 64×64 dummy PNG in /tmp (no big data file needed).
2.  Calls `infer.py` with that image.
3.  Asserts the script exits with code 0.

If you rename `infer.py` or change its CLI, tweak the subprocess call below.
"""
import subprocess, tempfile, pathlib

def test_infer_runs():
    tmpdir = pathlib.Path(tempfile.mkdtemp())
    dummy = tmpdir / "dummy.png"

    # Make a 64×64 grey square without pulling PIL if it isn't already there
    try:
        from PIL import Image, ImageDraw
        img = Image.new("L", (64, 64), 128)
        ImageDraw.Draw(img).text((4, 24), "OK", fill=255)
        img.save(dummy)
    except ModuleNotFoundError:
        # Pillow not present?  create an empty file – the script should still exit 0
        dummy.touch()

    out_png = tmpdir / "out.png"
    cmd = ["python", "infer.py", str(dummy), "--out", str(out_png)]
    result = subprocess.run(cmd, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()