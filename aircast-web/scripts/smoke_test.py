"""Responsive browser smoke test for the AirCast dashboard.

Run against an already-started dashboard, for example:
    uv run --group dev python aircast-web/scripts/smoke_test.py http://127.0.0.1:3100
"""

from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright


BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:3000"


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page_errors: list[str] = []
        page.on("pageerror", lambda error: page_errors.append(str(error)))

        response = page.goto(BASE_URL, wait_until="networkidle", timeout=60_000)
        assert response and response.status == 200, "Dashboard did not return HTTP 200"
        assert page.get_by_role("heading", name="Near-term PM₂.₅ outlook").is_visible()

        horizons = page.get_by_role("group", name="Live forecast")
        six_hour = horizons.get_by_role("button", name="6h")
        six_hour.click()
        page.wait_for_function(
            'document.querySelector("#live-pilot").innerText.includes("+6H")'
        )
        assert six_hour.get_attribute("aria-pressed") == "true"

        refresh = page.get_by_role("button", name="Refresh live air quality readings")
        refresh.click()
        page.wait_for_function(
            'document.querySelector("button[aria-label=\\"Refresh live air quality readings\\"]")?.disabled === false',
            timeout=15_000,
        )
        page.locator('nav[aria-label="Main navigation"] a[href="#network"]').click()
        assert page.evaluate("location.hash") == "#network"
        page.get_by_role("button", name="About forecast data").click()
        assert page.locator(".live-station-card").count() == 2

        for width, height in ((390, 844), (768, 1024), (1440, 960)):
            page.set_viewport_size({"width": width, "height": height})
            assert page.locator("body").evaluate(
                "element => element.scrollWidth === element.clientWidth"
            ), f"Horizontal overflow at viewport width {width}px"

        assert not page_errors, f"Browser JavaScript errors: {page_errors}"
        browser.close()


if __name__ == "__main__":
    main()
