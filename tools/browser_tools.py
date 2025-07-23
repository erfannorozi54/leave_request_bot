from typing import List, Optional, Dict, Any
import time
import json

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementNotInteractableException
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

    def take_screenshot(self) -> bytes:
        """Capture a PNG screenshot of the current page and return it as bytes."""
        screenshot_bytes = self.driver.get_screenshot_as_png()
        return screenshot_bytes

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

    def check_element_exists(self, selector: str) -> str:
        """Check if an element matching the CSS selector exists on the current page.
        
        Returns a JSON string with existence status and additional information.
        """
        try:
            element = self.driver.find_element(By.CSS_SELECTOR, selector)
            is_visible = element.is_displayed()
            is_enabled = element.is_enabled()
            tag_name = element.tag_name
            element_type = element.get_attribute("type") or "unknown"
            
            result = {
                "exists": True,
                "visible": is_visible,
                "enabled": is_enabled,
                "tag_name": tag_name,
                "type": element_type,
                "text": element.text[:100] if element.text else "",  # First 100 chars
                "message": f"Element '{selector}' found and is {'visible' if is_visible else 'hidden'}"
            }
            return json.dumps(result, indent=2)
        except NoSuchElementException:
            result = {
                "exists": False,
                "visible": False,
                "enabled": False,
                "message": f"Element '{selector}' not found on the page"
            }
            return json.dumps(result, indent=2)

    def find_elements_by_text(self, text: str, partial_match: bool = True) -> str:
        """Find all elements containing the specified text.
        
        Args:
            text: Text to search for
            partial_match: If True, search for partial matches; if False, exact matches only
        """
        try:
            if partial_match:
                elements = self.driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
            else:
                elements = self.driver.find_elements(By.XPATH, f"//*[text()='{text}']")
            
            results = []
            for i, element in enumerate(elements[:10]):  # Limit to first 10 results
                try:
                    results.append({
                        "index": i,
                        "tag_name": element.tag_name,
                        "text": element.text[:100],
                        "selector": self._generate_selector(element),
                        "visible": element.is_displayed(),
                        "enabled": element.is_enabled()
                    })
                except:
                    continue
            
            return json.dumps({
                "count": len(elements),
                "results": results,
                "message": f"Found {len(elements)} elements containing '{text}'"
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "count": 0,
                "results": [],
                "message": f"Error searching for text '{text}': {str(e)}"
            }, indent=2)

    def get_page_info(self) -> str:
        """Get comprehensive information about the current page including title, URL, and available elements."""
        try:
            page_info = {
                "title": self.driver.title,
                "url": self.driver.current_url,
                "elements": {}
            }
            
            # Count common element types
            common_selectors = {
                "buttons": "button, input[type='button'], input[type='submit'], input[type='reset']",
                "links": "a",
                "inputs": "input, textarea, select",
                "forms": "form",
                "images": "img",
                "tables": "table"
            }
            
            for element_type, selector in common_selectors.items():
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    page_info["elements"][element_type] = len(elements)
                except:
                    page_info["elements"][element_type] = 0
            
            return json.dumps(page_info, indent=2)
        except Exception as e:
            return json.dumps({
                "error": f"Failed to get page info: {str(e)}"
            }, indent=2)

    def safe_click_element(self, selector: str, timeout: int = 5) -> str:
        """Safely click an element with better error handling and waiting.
        
        This method waits for the element to be clickable before attempting to click.
        """
        try:
            # Wait for element to be present and clickable
            element = WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
            )
            element.click()
            return f"Successfully clicked element with selector: {selector}"
        except TimeoutException:
            return f"Element '{selector}' not clickable within {timeout} seconds"
        except ElementNotInteractableException:
            return f"Element '{selector}' is not interactable (may be hidden or disabled)"
        except NoSuchElementException:
            return f"Element '{selector}' not found on the page"
        except Exception as e:
            return f"Error clicking element '{selector}': {str(e)}"

    def safe_input_text(self, selector: str, text: str, timeout: int = 5) -> str:
        """Safely input text into an element with better error handling and waiting."""
        try:
            # Wait for element to be present and interactable
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
            
            # Check if element is interactable
            if not element.is_enabled():
                return f"Element '{selector}' is disabled and cannot receive input"
            
            element.clear()
            element.send_keys(text)
            return f"Successfully entered text '{text}' into element with selector: {selector}"
        except TimeoutException:
            return f"Element '{selector}' not found within {timeout} seconds"
        except ElementNotInteractableException:
            return f"Element '{selector}' is not interactable (may be hidden or disabled)"
        except NoSuchElementException:
            return f"Element '{selector}' not found on the page"
        except Exception as e:
            return f"Error entering text into element '{selector}': {str(e)}"

    def get_clickable_elements(self) -> str:
        """Get a list of all clickable elements on the current page with their selectors."""
        try:
            # Find all clickable elements
            clickable_selectors = [
                "button",
                "input[type='button']",
                "input[type='submit']",
                "input[type='reset']",
                "a[href]",
                "[onclick]",
                "[role='button']"
            ]
            
            all_clickable = []
            for selector in clickable_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if element.is_displayed() and element.is_enabled():
                            all_clickable.append({
                                "tag_name": element.tag_name,
                                "text": element.text[:50] if element.text else "",
                                "selector": self._generate_selector(element),
                                "type": element.get_attribute("type") or "unknown",
                                "href": element.get_attribute("href") or ""
                            })
                except:
                    continue
            
            # Remove duplicates based on selector
            unique_clickable = []
            seen_selectors = set()
            for element in all_clickable:
                if element["selector"] not in seen_selectors:
                    unique_clickable.append(element)
                    seen_selectors.add(element["selector"])
            
            return json.dumps({
                "count": len(unique_clickable),
                "elements": unique_clickable[:20],  # Limit to first 20
                "message": f"Found {len(unique_clickable)} clickable elements"
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "count": 0,
                "elements": [],
                "message": f"Error finding clickable elements: {str(e)}"
            }, indent=2)

    def get_form_elements(self) -> str:
        """Get a list of all form input elements on the current page."""
        try:
            form_selectors = [
                "input",
                "textarea",
                "select",
                "button[type='submit']"
            ]
            
            form_elements = []
            for selector in form_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if element.is_displayed():
                            form_elements.append({
                                "tag_name": element.tag_name,
                                "type": element.get_attribute("type") or "unknown",
                                "name": element.get_attribute("name") or "",
                                "id": element.get_attribute("id") or "",
                                "placeholder": element.get_attribute("placeholder") or "",
                                "value": element.get_attribute("value") or "",
                                "selector": self._generate_selector(element),
                                "required": element.get_attribute("required") is not None
                            })
                except:
                    continue
            
            return json.dumps({
                "count": len(form_elements),
                "elements": form_elements,
                "message": f"Found {len(form_elements)} form elements"
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "count": 0,
                "elements": [],
                "message": f"Error finding form elements: {str(e)}"
            }, indent=2)

    def wait_for_page_load(self, timeout: int = 10) -> str:
        """Wait for the page to fully load, including dynamic content."""
        try:
            # Wait for document ready state
            WebDriverWait(self.driver, timeout).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            
            # Additional wait for common dynamic content
            time.sleep(2)
            
            return f"Page loaded successfully within {timeout} seconds"
        except TimeoutException:
            return f"Page load timeout after {timeout} seconds"
        except Exception as e:
            return f"Error waiting for page load: {str(e)}"

    def refresh_page(self) -> str:
        """Refresh the current page and wait for it to load."""
        try:
            self.driver.refresh()
            self.wait_for_page_load()
            return "Page refreshed successfully"
        except Exception as e:
            return f"Error refreshing page: {str(e)}"

    def go_back(self) -> str:
        """Navigate back to the previous page in browser history."""
        try:
            self.driver.back()
            self.wait_for_page_load()
            return "Navigated back successfully"
        except Exception as e:
            return f"Error navigating back: {str(e)}"

    def go_forward(self) -> str:
        """Navigate forward in browser history."""
        try:
            self.driver.forward()
            self.wait_for_page_load()
            return "Navigated forward successfully"
        except Exception as e:
            return f"Error navigating forward: {str(e)}"

    def _generate_selector(self, element) -> str:
        """Generate a unique CSS selector for an element."""
        try:
            # Try ID first
            element_id = element.get_attribute("id")
            if element_id:
                return f"#{element_id}"
            
            # Try class
            element_class = element.get_attribute("class")
            if element_class:
                return f".{element_class.split()[0]}"  # Use first class
            
            # Fallback to tag name with position
            tag_name = element.tag_name
            parent = element.find_element(By.XPATH, "..")
            siblings = parent.find_elements(By.TAG_NAME, tag_name)
            for i, sibling in enumerate(siblings):
                if sibling == element:
                    return f"{tag_name}:nth-child({i + 1})"
            
            return tag_name
        except:
            return element.tag_name

    # ------------------------------------------------------------------
    # LangChain integration helpers
    # ------------------------------------------------------------------

    def get_tools(self) -> List[StructuredTool]:
        """Return every browser utility as a LangChain ``Tool`` instance."""
        return [
            StructuredTool.from_function(
                self.navigate_to_url,
                name="navigate_to_url",
                description="Navigate to a specific URL in the browser. Use this to go to any webpage. Always wait for the page to load after navigation.",
            ),
            StructuredTool.from_function(
                self.click_element,
                name="click_element",
                description="Click on the first element matching a CSS selector. Use this for basic clicking but prefer safe_click_element for better error handling.",
            ),
            StructuredTool.from_function(
                self.input_text,
                name="input_text",
                description="Type text into the element located by CSS selector. Use this for basic text input but prefer safe_input_text for better error handling.",
            ),
            StructuredTool.from_function(
                self.get_page_content,
                name="get_page_content",
                description="Return all visible text from the current page. Use this to understand what's currently displayed on the page.",
            ),
            StructuredTool.from_function(
                self.scroll,
                name="scroll",
                description="Scroll vertically by a specified number of pixels. Use positive numbers to scroll down, negative to scroll up. Useful for exploring long pages.",
            ),
            StructuredTool.from_function(
                self.press_key,
                name="press_key",
                description="Send a keyboard key (e.g. ENTER, TAB, ESC) to the element at selector. Useful for form submission or navigation.",
            ),
            StructuredTool.from_function(
                self.wait_for_element,
                name="wait_for_element",
                description="Wait until a CSS-selected element appears in the DOM. Use this when you expect an element to load dynamically.",
            ),
            StructuredTool.from_function(
                self.get_element_text,
                name="get_element_text",
                description="Extract the inner text of a CSS-selected element. Use this to get specific text content from elements.",
            ),
            StructuredTool.from_function(
                self.take_screenshot,
                name="take_screenshot",
                description="Capture a PNG screenshot of the current page and return it as bytes for LLM processing. Use this to see the visual state of the page.",
            ),
            StructuredTool.from_function(
                self.open_new_tab,
                name="open_new_tab",
                description="Open a new browser tab and optionally navigate to a URL. Useful for opening links without losing the current page.",
            ),
            StructuredTool.from_function(
                self.switch_tab,
                name="switch_tab",
                description="Switch browser context to the tab at a given index (0-based). Use this to work with multiple tabs.",
            ),
            StructuredTool.from_function(
                self.close_current_tab,
                name="close_current_tab",
                description="Close the active browser tab and switch to the last remaining tab. Use this to clean up tabs.",
            ),
            StructuredTool.from_function(
                self.close_browser,
                name="close_browser",
                description="Completely shut down the browser instance. Use this when you're done with all browser operations.",
            ),
            # New enhanced tools
            StructuredTool.from_function(
                self.check_element_exists,
                name="check_element_exists",
                description="Check if an element matching the CSS selector exists on the current page. Returns detailed JSON with existence status, visibility, and element properties. Use this BEFORE trying to interact with elements to avoid errors.",
            ),
            StructuredTool.from_function(
                self.find_elements_by_text,
                name="find_elements_by_text",
                description="Find all elements containing the specified text on the current page. Returns JSON with element details and selectors. Use this to discover elements when you know the text but not the selector. Set partial_match=True for flexible matching.",
            ),
            StructuredTool.from_function(
                self.get_page_info,
                name="get_page_info",
                description="Get comprehensive information about the current page including title, URL, and counts of different element types (buttons, links, inputs, etc.). Use this to understand the page structure and available interactive elements.",
            ),
            StructuredTool.from_function(
                self.safe_click_element,
                name="safe_click_element",
                description="Safely click an element with better error handling and waiting. Waits for the element to be clickable before attempting to click. Returns detailed error messages if the element is not found, not clickable, or not interactable. Use this instead of click_element for more reliable interactions.",
            ),
            StructuredTool.from_function(
                self.safe_input_text,
                name="safe_input_text",
                description="Safely input text into an element with better error handling and waiting. Checks if the element is enabled before attempting to input. Returns detailed error messages if the element is not found or not interactable. Use this instead of input_text for more reliable interactions.",
            ),
            StructuredTool.from_function(
                self.get_clickable_elements,
                name="get_clickable_elements",
                description="Get a list of all clickable elements on the current page with their selectors, text, and properties. Returns JSON with up to 20 clickable elements. Use this to discover what can be clicked on the page when you're not sure what elements are available.",
            ),
            StructuredTool.from_function(
                self.get_form_elements,
                name="get_form_elements",
                description="Get a list of all form input elements on the current page including inputs, textareas, selects, and submit buttons. Returns JSON with element details like type, name, placeholder, and whether they're required. Use this to understand form structure before filling it out.",
            ),
            StructuredTool.from_function(
                self.wait_for_page_load,
                name="wait_for_page_load",
                description="Wait for the page to fully load, including dynamic content. Use this after navigation or actions that might trigger page changes to ensure the page is ready for interaction.",
            ),
            StructuredTool.from_function(
                self.refresh_page,
                name="refresh_page",
                description="Refresh the current page and wait for it to load. Use this when you need to reload the page content or if the page seems stuck.",
            ),
            StructuredTool.from_function(
                self.go_back,
                name="go_back",
                description="Navigate back to the previous page in browser history. Use this to return to the previous page without losing your place.",
            ),
            StructuredTool.from_function(
                self.go_forward,
                name="go_forward",
                description="Navigate forward in browser history. Use this to go forward after using go_back.",
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
