"""Screenshot widgets for review: python xp/shot.py OUT_DIR name1 name2 ... (serves xp/ on a local port).
Each widget is captured as OUT_DIR/<name>.png after a short settle time. Console errors are printed."""
import sys, threading, http.server, functools, pathlib, socketserver
from playwright.sync_api import sync_playwright

here = pathlib.Path(__file__).resolve().parent
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
names = sys.argv[2:]
class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
handler = functools.partial(Quiet, directory=str(here))
srv = socketserver.TCPServer(("127.0.0.1", 0), handler)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    b = p.chromium.launch()
    for name in names:
        base, _, extra = name.partition("?")
        pg = b.new_page(viewport={"width": 1920, "height": 1080})
        pg.on("console", lambda m: print(f"[{base}] {m.type}: {m.text}") if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: print(f"[{base}] pageerror: {e}"))
        pg.goto(f"http://127.0.0.1:{port}/dev.html?w={base}" + (f"&{extra}" if extra else ""))
        pg.wait_for_selector("body[data-ready='1']", timeout=60000)
        pg.wait_for_timeout(1500)
        tag = base + ("_" + extra.replace("=", "-").replace("&", "_") if extra else "")
        pg.locator(".frame").first.screenshot(path=str(out / f"{tag}.png"))
        print("saved", out / f"{tag}.png")
        pg.close()
    b.close()
srv.shutdown()
