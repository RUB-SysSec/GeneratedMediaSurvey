import random
import re
import time
from typing import Dict, List, Optional

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from test_utils import TEST_RAND_CHOICES, FlaskUnsigner, TestConfig


def chrome_driver(mobile: bool = False) -> webdriver.Chrome:
    """Returns an initialized selenium chrome client.
    """
    opts = Options()
    opts.add_argument("headless")
    if mobile:
        # TODO: replace with mobile emulation when it actually can click buttons
        # mobile_emulation = {"deviceName": "iPhone SE"}
        # opts.add_experimental_option("mobileEmulation", mobile_emulation)
        opts.add_argument("window-size=375,667")  # Iphone SE
    else:
        opts.add_argument("window-size=1920,1080")
    return webdriver.Chrome(options=opts)


class ChromeClient:
    """A Wrapper class for a selenium chrome client.
    """

    def __init__(
        self,
        mobile: bool = False,
        timeout: int = 5,
        wait: float = 1.5,
    ) -> None:
        self.mobile = mobile
        self.timeout = timeout
        self.wait_amount = wait
        self.elements_choosen: List[str] = []
        self.pass_attention = False

        self._serializer = FlaskUnsigner().get_signing_serializer(
            TestConfig.SECRET_KEY)
        self._driver = chrome_driver(mobile=self.mobile)

    def accept_conditions(self):
        """Accept initial conditions.
        """
        self.click("continue")
        self.click("consentCheckboxAccept")
        self.click("accept_conditions")
        self.wait_for_url_change()

    def perform_experiment(self, steps: int):
        """Perform the entire experiment.
        """
        for _ in range(steps):
            self.perform_experiment_step()

    def perform_experiment_step(self):
        """One step of the experiemnt.
        """
        try:
            element = self._driver.find_element(By.ID, "scale")
            self._scale_experiment(element)
        except NoSuchElementException:
            self._choice_experimet()

        self.submit()
        time.sleep(self.wait_amount)

    def _scale_experiment(self, element: WebElement):
        value = random.randint(0, 100)
        self._driver.execute_script(
            "arguments[0].value = arguments[1]", element, value)
        self.elements_choosen.append(value)

    def _choice_experimet(self):
        choices = self._driver.find_elements(
            By.CLASS_NAME, "option-label")

        choice = random.choice(choices)
        idx = choice.text
        match = re.search("-?[0-9]+", idx)
        if match:
            idx_val = int(match[0])
        else:
            idx_val = idx
        self.elements_choosen.append(idx_val)
        self.click_element(choice)

    def questionnaire(self):
        """Solve questionnaire.
        """
        url = self.current_url
        while self.current_url == url:
            # collect categories
            category = self._driver.find_elements(By.CLASS_NAME, "category")[0]

            # pick random options
            self.pick_single_category(category)

            # move forwards
            self.click("continue")

            self._assert_logs()
            time.sleep(self.wait_amount)

    def pick_single_category(self, cat: WebElement):
        """Pick all elements for a single questionnaire category.
        """
        # pick random options
        for question in cat.find_elements(By.CLASS_NAME, "question"):
            # choose option
            question_type = question.get_attribute("data-type")
            if question_type == "likert" \
                    or question_type == "options" \
                    or question_type == "education":
                options = question.find_elements(By.TAG_NAME, "label")

                choice = random.choice(options)
                self.click_element(choice)

                # get input and extract data
                for_input = choice.get_attribute("for")
                element = self._driver.find_element_by_id(for_input)
                idx = element.get_attribute("data-option-id")
                self.elements_choosen.append(idx)

            elif question_type == "attention":
                possible_options = question.find_elements(
                    By.TAG_NAME, "label")
                options = [re.sub("(\n|\+|-|[0-9])", "", ele.text).strip()
                           for ele in possible_options]

                # build regex on-the-fly
                correct_re = f"({'|'.join(options)})"
                question_text = question.text
                correct_match = re.search(correct_re, question_text)
                assert correct_match

                correct = correct_match[1]

                for option in possible_options:
                    if (option.text == correct and self.pass_attention) \
                            or (option.text != correct and not self.pass_attention):
                        self.click_element(option)
                        break

            elif question_type == "scale" or question_type == "number" or question_type == "age":
                if question_type == "age":
                    value = random.randint(18, 100)
                else:
                    value = random.randint(0, 100)

                scale = question.find_element(By.TAG_NAME, "input")
                self._driver.execute_script(
                    "arguments[0].value = arguments[1]", scale, value)
                self.elements_choosen.append(str(value))

            elif question_type == "textfield":
                textarea = question.find_element(By.TAG_NAME, "textarea")
                value = random.choice(TEST_RAND_CHOICES)
                self._driver.execute_script(
                    "arguments[0].value = arguments[1]", textarea, value)
                self.elements_choosen.append(value)
            else:
                raise NotImplementedError(
                    f"Question type not supported: {question_type}")

    def get(self, url: str):
        """Get url and wait till new page is loaded.
        """
        self._driver.get(url)
        self._driver.implicitly_wait(5)

    def submit(self, check_error: bool = True):
        """Click the #sbumit button.
        """
        self.click("submit")
        if check_error:
            self._assert_logs()

    def click(self, idx: str):
        """Click the element with idx.
        """
        element = self._driver.find_element(By.ID, idx)
        self.click_element(element)

    def click_element(self, element: WebElement):
        """Click some element.
        """
        self._driver.execute_script(
            "arguments[0].scrollIntoView({block: 'center', inline: 'nearest'})", element)
        time.sleep(self.wait_amount)
        element.click()

    def quit(self):
        """Shutdown driver.
        """
        self._driver.quit()

    def wait_for_url_change(self):
        """Wait for url change.
        """
        current_url = self._driver.current_url
        WebDriverWait(self._driver, self.timeout).until(
            EC.url_changes(current_url))

    def _assert_logs(self):
        for log in self._driver.get_log('browser'):
            if "favicon" in log["message"]:
                continue

            assert False, log["message"]

    @ property
    def current_url(self) -> str:
        """Returns the current url of the driver.
        """
        return self._driver.current_url

    @property
    def html(self) -> str:
        """Returns the html of the current page.
        """
        return self._driver.page_source

    def is_displayed(self, idx: str) -> bool:
        """Check if elements is displayed
        """
        return self._driver.find_element(By.ID, idx).is_displayed()

    @property
    def cookie(self) -> Optional[Dict]:
        """Return the current cookie values as Dict (only first cookie).
        Returns None when there are no cookies.
        """
        cookies = self._driver.get_cookies()
        if len(cookies) == 0:
            return None
        value = cookies[0]["value"]

        return self._serializer.loads(value)
