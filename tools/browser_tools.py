from typing import List, Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain.tools import StructuredTool


class BrowserTools:
    """Collection of Selenium-powered browser utilities exposed as LangChain tools."""

    def __init__(
        self,
        driver: Optional[webdriver.Chrome] = None,
        headless: bool = False,
        page_load_timeout: int = 30,
    ) -> None:
        """Create a BrowserTools instance.

        Args:
            driver: Pre-configured Selenium WebDriver. If None, a new Chrome driver
                will be created automatically.
            headless: Whether to launch the browser in headless mode (default: True).
            page_load_timeout: Seconds to wait before a page-load is considered timed-out.
        """
        if driver:
            self.driver = driver
        else:
            options = Options()
            if headless:
                # Chrome 109+ supports --headless=new which is more stable.
                options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            self.driver = webdriver.Chrome(options=options)

        # Fail fast if a page takes too long to load
        self.driver.set_page_load_timeout(page_load_timeout)

    # ---------------------------------------------------------------------
    # Tool implementations
    # ---------------------------------------------------------------------

    def navigate_to_url(self, url: str) -> str:
        """Navigate the browser to ``url``."""
        self.driver.get(url)
        return f"Navigated to {url}"

    def click_element(self, selector: str) -> str:
        """Click the first element matching the given CSS ``selector``."""
        element = self.driver.find_element(By.CSS_SELECTOR, selector)
        element.click()
        return f"Clicked element with selector: {selector}"

    def input_text(self, selector: str, text: str) -> str:
        """Type ``text`` into the element located by CSS ``selector``."""
        element = self.driver.find_element(By.CSS_SELECTOR, selector)
        element.clear()
        element.send_keys(text)
        return f"Entered text '{text}' into field with selector: {selector}"

    def get_page_content(self) -> str:
        """Return all visible text from the current page."""
        return self.driver.find_element(By.TAG_NAME, "body").text

    def scroll(self, pixels: int = 1000) -> str:
        """Scroll vertically by ``pixels`` (positive => down, negative => up)."""
        self.driver.execute_script("window.scrollBy(0, arguments[0]);", pixels)
        return f"Scrolled page by {pixels} pixels"

    def press_key(self, selector: str, key: str) -> str:
        """Send a keyboard ``key`` (e.g. ENTER, TAB) to the element at ``selector``."""
        key_map = {
            "ENTER": Keys.ENTER,
            "TAB": Keys.TAB,
            "ESC": Keys.ESCAPE,
            "ESCAPE": Keys.ESCAPE,
            "SPACE": Keys.SPACE,
            "BACKSPACE": Keys.BACKSPACE,
        }
        if key.upper() not in key_map:
            return f"Unsupported key '{key}'. Supported keys: {', '.join(key_map)}"
        element = self.driver.find_element(By.CSS_SELECTOR, selector)
        element.send_keys(key_map[key.upper()])
        return f"Pressed {key.upper()} on element with selector: {selector}"

    def wait_for_element(self, selector: str, timeout: int = 10) -> str:
        """Block execution until an element matching ``selector`` is present."""
        WebDriverWait(self.driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )
        return f"Element with selector '{selector}' appeared within {timeout} s"

    def get_element_text(self, selector: str) -> str:
        """Return the text content of the element specified by ``selector``."""
        element = self.driver.find_element(By.CSS_SELECTOR, selector)
        return element.text

    def take_screenshot(self, file_path: str = "screenshot.png") -> str:
        """Save a PNG screenshot of the current page to ``file_path``."""
        self.driver.save_screenshot(file_path)
        return f"Screenshot saved to {file_path}"

    def open_new_tab(self, url: Optional[str] = None) -> str:
        """Open a new browser tab and optionally navigate to ``url``."""
        self.driver.execute_script("window.open('');")
        self.driver.switch_to.window(self.driver.window_handles[-1])
        if url:
            self.driver.get(url)
            return f"Opened new tab and navigated to {url}"
        return "Opened new blank tab"

    def switch_tab(self, index: int = 0) -> str:
        """Switch to the tab at ``index`` (0-based)."""
        handles = self.driver.window_handles
        if index < 0 or index >= len(handles):
            raise IndexError(
                f"Tab index {index} out of range. {len(handles)} tab(s) open."
            )
        self.driver.switch_to.window(handles[index])
        return f"Switched to tab {index}"

    def close_current_tab(self) -> str:
        """Close the active tab and switch to the last remaining handle."""
        self.driver.close()
        if self.driver.window_handles:
            self.driver.switch_to.window(self.driver.window_handles[-1])
        return "Closed current tab"

    def close_browser(self) -> str:
        """Terminate the browser session entirely."""
        self.driver.quit()
        return "Browser closed"

    # ------------------------------------------------------------------
    # LangChain integration helpers
    # ------------------------------------------------------------------

    def get_tools(self) -> List[StructuredTool]:
        """Return every browser utility as a LangChain ``Tool`` instance."""
        return [
            StructuredTool.from_function(
                self.navigate_to_url,
                name="navigate_to_url",
                description="Navigate to a specific URL in the browser.",
            ),
            StructuredTool.from_function(
                self.click_element,
                name="click_element",
                description="Click on the first element matching a CSS selector.",
            ),
            StructuredTool.from_function(
                self.input_text,
                name="input_text",
                description="Type text into the element located by CSS selector.",
            ),
            StructuredTool.from_function(
                self.get_page_content,
                name="get_page_content",
                description="Return all visible text from the current page.",
            ),
            StructuredTool.from_function(
                self.scroll,
                name="scroll",
                description="Scroll vertically by a specified number of pixels.",
            ),
            StructuredTool.from_function(
                self.press_key,
                name="press_key",
                description="Send a keyboard key (e.g. ENTER, TAB) to the element at selector.",
            ),
            StructuredTool.from_function(
                self.wait_for_element,
                name="wait_for_element",
                description="Wait until a CSS-selected element appears in the DOM.",
            ),
            StructuredTool.from_function(
                self.get_element_text,
                name="get_element_text",
                description="Extract the inner text of a CSS-selected element.",
            ),
            StructuredTool.from_function(
                self.take_screenshot,
                name="take_screenshot",
                description="Capture a PNG screenshot of the current page.",
            ),
            StructuredTool.from_function(
                self.open_new_tab,
                name="open_new_tab",
                description="Open a new browser tab and optionally navigate to a URL.",
            ),
            StructuredTool.from_function(
                self.switch_tab,
                name="switch_tab",
                description="Switch browser context to the tab at a given index.",
            ),
            StructuredTool.from_function(
                self.close_current_tab,
                name="close_current_tab",
                description="Close the active browser tab.",
            ),
            StructuredTool.from_function(
                self.close_browser,
                name="close_browser",
                description="Completely shut down the browser instance.",
            ),
        ]


# Convenience helper — allows direct import without instantiating the class.
# Example usage:
#     from tools.browser_tools import get_tools
#     tools = get_tools()


def get_tools(**kwargs) -> List[StructuredTool]:  # pylint: disable=invalid-name
    """Module-level wrapper that instantiates :class:`BrowserTools` and returns its tools."""
    return BrowserTools(**kwargs).get_tools()


if __name__ == "__main__":
    tools = get_tools()
    tool = tools[0]
    print(tool.name)
    print(tool.args)
