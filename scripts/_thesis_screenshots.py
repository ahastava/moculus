"""Capture thesis figures from the running MoCoLUS web UI via Playwright."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

URL = "http://localhost:8080"
OUT = Path("thesis_figures")


async def select_scenario(page, key: str):
    await page.evaluate(
        """([k]) => {
            const sel = document.getElementById('scenario-select');
            sel.value = k;
            sel.dispatchEvent(new Event('change', {bubbles: true}));
        }""",
        [key],
    )
    await page.wait_for_timeout(2500)


async def click_probe_canvas_at_zone(page, zone_key: str):
    """Snap probe to a known zone by dispatching mousedown/up on probe-canvas
    at the zone's stored normalized coordinates."""
    await page.evaluate(
        """([z]) => {
            // probeOverlay is locally scoped; reach the canvas + bounds
            const cv = document.getElementById('probe-canvas');
            // The renderer resizes the backing canvas to clientWidth*dpr.
            // Click events use clientX/Y in CSS px, anchored to the bounding rect.
            const rect = cv.getBoundingClientRect();
            // Patient's left = viewer's right on standard anatomical chest map.
            const positions = {
                'UPPER_BLUE_L': [0.70, 0.25],
                'UPPER_BLUE_R': [0.30, 0.25],
                'LOWER_BLUE_L': [0.70, 0.55],
                'LOWER_BLUE_R': [0.30, 0.55],
            };
            const p = positions[z] || [0.30, 0.25];
            const x = rect.left + p[0] * rect.width;
            const y = rect.top + p[1] * rect.height;
            const opts = {bubbles: true, clientX: x, clientY: y, button: 0};
            cv.dispatchEvent(new MouseEvent('mousedown', opts));
            cv.dispatchEvent(new MouseEvent('mouseup', opts));
        }""",
        [zone_key],
    )
    await page.wait_for_timeout(2500)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1600, "height": 1000})
        page = await ctx.new_page()

        page.on("console", lambda msg: print(f"[console.{msg.type}] {msg.text}"))
        page.on("pageerror", lambda exc: print(f"[pageerror] {exc}"))

        await page.goto(URL, wait_until="networkidle")
        await page.wait_for_function(
            "document.getElementById('scenario-select') && "
            "document.getElementById('scenario-select').options.length > 0",
            timeout=15000,
        )
        await page.wait_for_timeout(2000)

        # ---- Tension Pneumothorax (Left) ----
        await select_scenario(page, "left_pneumothorax")
        await click_probe_canvas_at_zone(page, "UPPER_BLUE_L")

        await page.screenshot(path=str(OUT / "fig_3_9_web_interface.png"), full_page=True)
        print("wrote fig_3_9_web_interface.png")

        bmode = page.locator("#bmode-canvas").first
        mmode = page.locator("#mmode-canvas").first
        try:
            await bmode.scroll_into_view_if_needed()
            box_b = await bmode.bounding_box()
            box_m = await mmode.bounding_box() if await mmode.count() else None
            if box_b:
                if box_m:
                    x0 = min(box_b["x"], box_m["x"]) - 10
                    y0 = min(box_b["y"], box_m["y"]) - 10
                    x1 = max(box_b["x"] + box_b["width"], box_m["x"] + box_m["width"]) + 10
                    y1 = max(box_b["y"] + box_b["height"], box_m["y"] + box_m["height"]) + 10
                else:
                    x0, y0 = box_b["x"] - 10, box_b["y"] - 10
                    x1, y1 = box_b["x"] + box_b["width"] + 10, box_b["y"] + box_b["height"] + 10
                clip = {"x": max(0, x0), "y": max(0, y0),
                        "width": x1 - x0, "height": y1 - y0}
                await page.screenshot(path=str(OUT / "web_bmode_crop.png"), clip=clip)
                print(f"wrote web_bmode_crop.png clip={clip}")
        except Exception as e:
            print(f"bmode crop failed: {e}")

        # Probe / zone overlay panel
        try:
            probe = page.locator("#probe-canvas").first
            if await probe.count():
                await probe.scroll_into_view_if_needed()
                box = await probe.bounding_box()
                if box:
                    pad = 20
                    clip = {"x": max(0, box["x"] - pad),
                            "y": max(0, box["y"] - pad),
                            "width": box["width"] + 2 * pad,
                            "height": box["height"] + 2 * pad}
                    await page.screenshot(path=str(OUT / "web_zone_overlay.png"), clip=clip)
                    print(f"wrote web_zone_overlay.png clip={clip}")
        except Exception as e:
            print(f"zone overlay crop failed: {e}")

        # ---- Normal scenario ----
        await select_scenario(page, "normal")
        await click_probe_canvas_at_zone(page, "UPPER_BLUE_L")
        await page.screenshot(path=str(OUT / "web_normal_scenario.png"), full_page=True)
        print("wrote web_normal_scenario.png")

        await ctx.close()
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
