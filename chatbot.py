from dotenv import load_dotenv
import streamlit as st
import PyPDF2
from PIL import Image
import requests
import base64
import os
import io
import csv

# ---------- LOAD ENV ----------
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_KEY:
    st.error("❌ GROQ_API_KEY missing. Check .env file")
    st.stop()

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="QA Test Assistant",
    page_icon="🧪",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
        padding: 20px; border-radius: 10px;
        text-align: center; margin-bottom: 20px;
    }
    .main-header h1 { color: #e94560; font-size: 2rem; margin: 0; }
    .main-header p { color: #a8b2d8; margin: 5px 0 0 0; }
    .metric-card {
        background: #f8f9fa; border-radius: 8px;
        padding: 16px; text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-num { font-size: 32px; font-weight: 700; }
    .metric-lbl { font-size: 13px; color: #444 !important; margin-top: 4px; }
    .badge-high { background:#ffe0e0; color:#c00;
        padding:2px 8px; border-radius:99px; font-size:11px; }
    .badge-med { background:#fff3cd; color:#856404;
        padding:2px 8px; border-radius:99px; font-size:11px; }
    .badge-low { background:#d4edda; color:#155724;
        padding:2px 8px; border-radius:99px; font-size:11px; }
    .badge-pos { background:#d1f5ea; color:#0f6e56;
        padding:2px 8px; border-radius:99px; font-size:11px; }
    .badge-neg { background:#fde8e8; color:#a32d2d;
        padding:2px 8px; border-radius:99px; font-size:11px; }
    .ai-suggestion {
        background: #e8f4fd; border-left: 4px solid #185FA5;
        padding: 10px 14px; border-radius: 0 8px 8px 0;
        font-size: 14px; margin: 8px 0;
        color: #0a2540 !important;
        font-weight: 500 !important;
    }
    .duplicate-alert {
        background: #fff3cd; border-left: 4px solid #ffc107;
        padding: 10px 14px; border-radius: 0 8px 8px 0;
        font-size: 14px; margin: 8px 0;
        color: #4a3800 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🧪 QA Test Assistant</h1>
    <p>AI-Powered Test Case Generator | Selenium Code | Coverage Analyzer</p>
</div>
""", unsafe_allow_html=True)

# ✅ Auto scroll to bottom
st.markdown("""
<script>
    window.addEventListener('load', function() {
        setTimeout(function() {
            window.scrollTo(0, document.body.scrollHeight);
        }, 500);
    });
</script>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "images" not in st.session_state:
    st.session_state.images = []
if "file_text" not in st.session_state:
    st.session_state.file_text = ""
if "prev_file_names" not in st.session_state:
    st.session_state.prev_file_names = []
if "last_test_cases" not in st.session_state:
    st.session_state.last_test_cases = []
if "last_ac" not in st.session_state:
    st.session_state.last_ac = ""
if "dashboard_data" not in st.session_state:
    st.session_state.dashboard_data = None
if "last_reply" not in st.session_state:
    st.session_state.last_reply = ""
if "last_feature" not in st.session_state:
    st.session_state.last_feature = ""
if "last_file_prefix" not in st.session_state:
    st.session_state.last_file_prefix = ""
if "last_action" not in st.session_state:
    st.session_state.last_action = ""


# ===============================
# 🔧 HELPERS
# ===============================
def image_to_base64(image: Image.Image) -> str:
    image = image.convert("RGB")
    image.thumbnail((1024, 1024))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_groq(messages: list, images: list = None) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json",
    }
    if images:
        model = "meta-llama/llama-4-scout-17b-16e-instruct"
        api_messages = []
        for i, msg in enumerate(messages):
            if i == len(messages) - 1 and msg["role"] == "user":
                content = [{"type": "text", "text": msg["content"]}]
                for img in images:
                    b64 = image_to_base64(img)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        }
                    })
                api_messages.append(
                    {"role": "user", "content": content})
            else:
                api_messages.append(msg)
    else:
        model = "llama-3.3-70b-versatile"
        api_messages = messages

    payload = {
        "model": model,
        "messages": api_messages,
        "max_tokens": 7000,
        "temperature": 0.3,
    }
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload, timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "❌ Request timed out. Please try again."
    except requests.exceptions.HTTPError:
        return f"❌ API error {response.status_code}: {response.text}"
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"


# ===============================
# 📊 DASHBOARD FUNCTIONS
# ===============================
def compute_dashboard(test_cases: list, ac_text: str) -> dict:

    unique_titles = list(dict.fromkeys(
        tc["Test Case Title"] for tc in test_cases
        if tc.get("Test Case Title")
    ))

    positive = sum(
        1 for title in unique_titles
        if "not able" not in title.lower()
        and "should not" not in title.lower()
        and "unable" not in title.lower()
        and "invalid" not in title.lower()
    )

    negative = sum(
        1 for title in unique_titles
        if "not able" in title.lower()
        or "should not" in title.lower()
        or "unable" in title.lower()
        or "invalid" in title.lower()
    )

    high = sum(
        1 for t in unique_titles
        if any(word in t.lower() for word in [
            "login", "security", "payment",
            "critical", "error", "not able",
            "invalid", "fail", "unable"
        ])
    )
    med = max(0, len(unique_titles) - high)

    ac_lines = [
        l.strip() for l in ac_text.split("\n")
        if l.strip() and len(l.strip()) > 10
    ]
    covered = 0
    for ac_line in ac_lines:
        keywords = [
            w for w in ac_line.lower().split()
            if len(w) > 4
        ][:3]
        for tc in test_cases:
            tc_text = (
                tc.get("Steps to Reproduce", "") +
                tc.get("Test Case Title", "")
            ).lower()
            if any(kw in tc_text for kw in keywords):
                covered += 1
                break
    coverage_pct = int(
        (covered / max(len(ac_lines), 1)) * 100
    )

    all_titles = [
        tc.get("Test Case Title", "")
        for tc in test_cases
    ]
    duplicates = len(all_titles) - len(set(all_titles))

    return {
        "total": len(unique_titles),
        "positive": positive,
        "negative": negative,
        "unique_tcs": len(unique_titles),
        "high": high,
        "med": med,
        "coverage_pct": coverage_pct,
        "duplicates": duplicates,
        "unique_titles": unique_titles[:8]
    }


def generate_suggestions(data: dict) -> list:
    suggestions = []
    if data['negative'] == 0:
        suggestions.append(
            "⚠️ No negative test cases found. "
            "Add error handling and invalid input scenarios.")
    if data['coverage_pct'] < 80:
        suggestions.append(
            f"⚠️ Coverage is {data['coverage_pct']}%. "
            "Some AC points may not be fully covered.")
    if data['total'] < 5:
        suggestions.append(
            "💡 Only a few steps generated. "
            "Consider adding more edge case scenarios.")
    if not suggestions:
        suggestions.append(
            "✅ Great coverage! All positive and negative "
            "scenarios are well covered. "
            "Test suite looks comprehensive.")
    return suggestions


def show_dashboard(data: dict, feature: str):
    st.markdown("---")
    st.markdown(f"### 📊 Live Dashboard — *{feature}*")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-num" style="color:#185FA5">
                {data['unique_tcs']}
            </div>
            <div class="metric-lbl">Total Test Cases</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-num" style="color:#3B6D11">
                {data['positive']}
            </div>
            <div class="metric-lbl">Positive Test Cases</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-num" style="color:#A32D2D">
                {data['negative']}
            </div>
            <div class="metric-lbl">Negative Test Cases</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        color = (
            "#3B6D11" if data['coverage_pct'] >= 80
            else "#854F0B" if data['coverage_pct'] >= 50
            else "#A32D2D"
        )
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-num" style="color:{color}">
                {data['coverage_pct']}%
            </div>
            <div class="metric-lbl">AC Coverage</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🎯 Coverage Breakdown")
        pos_pct = int(
            (data['positive'] / max(data['total'], 1)) * 100)
        neg_pct = int(
            (data['negative'] / max(data['total'], 1)) * 100)
        st.markdown(f"**Positive cases** — {pos_pct}%")
        st.progress(pos_pct / 100)
        st.markdown(f"**Negative cases** — {neg_pct}%")
        st.progress(neg_pct / 100)
        st.markdown(f"**AC Coverage** — {data['coverage_pct']}%")
        st.progress(data['coverage_pct'] / 100)
        st.markdown("#### 🏷️ Priority Distribution")
        p1, p2 = st.columns(2)
        with p1:
            st.markdown(
                f'<span class="badge-high">'
                f'🔴 High: {data["high"]}</span>',
                unsafe_allow_html=True)
        with p2:
            st.markdown(
                f'<span class="badge-med">'
                f'🟡 Medium: {data["med"]}</span>',
                unsafe_allow_html=True)

    with col_right:
        st.markdown("#### 🔍 AI Duplicate Detector")
        if data['duplicates'] == 0:
            st.markdown("""
            <div class="ai-suggestion">
                ✅ No duplicate test cases found.
                All test cases cover unique scenarios.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="duplicate-alert">
                ⚠️ {data['duplicates']} potential duplicate(s) detected.
                Review and consolidate for cleaner test suite.
            </div>""", unsafe_allow_html=True)

        st.markdown("#### 💡 AI Smart Suggestions")
        for s in generate_suggestions(data):
            st.markdown(
                f'<div class="ai-suggestion">{s}</div>',
                unsafe_allow_html=True)

    st.markdown("#### 📋 Test Cases Generated")
    for i, title in enumerate(data['unique_titles'], 1):
        ptype = "neg" if "not" in title.lower() else "pos"
        label = "Negative" if ptype == "neg" else "Positive"
        st.markdown(
            f'{i}. <span class="badge-{ptype}">{label}</span> {title}',
            unsafe_allow_html=True)


# ===============================
# 📋 PROMPT TEMPLATES
# ===============================

INTERMEDIATE_STEPS_INSTRUCTION = """
IMPORTANT: After step 7 user is ALREADY LOGGED IN.
Do NOT repeat any login steps again after step 7.
Add 2-3 navigation steps like:
- "Navigate to the Product Listing Page (PLP)","User should be able to view the PLP page","User is able to view the PLP page"
- "Select the product subcategory from navigation","User should be able to view products under selected category","User is able to view products under selected category"
- "Click on the relevant section or tab","User should be able to view the section","User is able to view the section"
- "Select a product from the list","User should be able to view product details","User is able to view product details"
Then add the final verification step based on AC.
DO NOT jump from step 7 directly to final verification.
"""


def get_testcase_prompt(
        ac_text: str,
        feature_name: str = "Feature") -> str:
    return f"""You are a Technical Test Lead. Generate test cases for:

Feature: {feature_name}

Acceptance Criteria:
{ac_text}

EVERY test case MUST start with these 7 login steps in CSV:
"Launch the following url https://t1-aeg-qa-a.eluxmkt.com/der/de/b2b/pre-login/","User should be able to launch the url","User is able to launch the url"
"Click on the Partner link from the portal","User should be able to click on the partner link","User is able to Click on the partner link"
"Verify that the user is redirected to Prelogin page","User should be able to view the Prelogin page","User is able to View the Prelogin page"
"Verify whether the user is able to see Login now and Contact us Buttons","User should be able to see Login now and Contact us Buttons","User is able to View the Login now and Contact us Buttons"
"Click on Login now from prelogin page","User should be able to click Login now button","User is able to Click on the login now button"
"Verify it is redirected to SAML login page","User should be able to view the SAML login page","User is able to view the SAML login page"
"Enter user credentials and login","User should be able to login and view Chiron home page","User is able to View the Chiron home page"

{INTERMEDIATE_STEPS_INSTRUCTION}

Generate 6-8 test cases (mix positive and negative).

First show titles only in chat:
✅ Generated Test Cases:
1. Verify whether user is able to [title]
2. Verify whether user is not able to [title]

Then IMMEDIATELY provide the full CSV — do not skip:
---CSV START---
Test Case Title,Steps to Reproduce,Expected Result,Actual Result
---CSV END---

CSV Rules — 4 COLUMNS:
Column 1 - Test Case Title: same title repeats for every step
Column 2 - Steps to Reproduce: ACTION (Launch/Click/Navigate/Enter/Verify)
Column 3 - Expected Result:
  POSITIVE = "User should be able to [action]"
  NEGATIVE = "User should not be able to [action]"
Column 4 - Actual Result:
  POSITIVE = "User is able to [action]"
  NEGATIVE = "User is not able to [action]"
- Include all 7 login steps for every TC
- Then 2-3 navigation steps (no login repeat!)
- Then final verification step
- DO NOT skip the CSV section!"""


def get_selenium_prompt(
        ac_text: str, tc_text: str = "") -> str:
    return f"""Act as a Senior Selenium Automation Engineer.

Generate complete Selenium Java code using:
- TestNG framework
- Page Object Model (POM)
- WebDriverManager for driver setup
- TestNG Assert for assertions

Acceptance Criteria:
{ac_text}

{f"Test Cases context:{chr(10)}{tc_text}" if tc_text else ""}

Generate:
1. Page Object class with @FindBy WebElements and methods
2. TestNG Test class with @BeforeClass @Test @AfterClass
3. Both positive and negative test methods
4. Login flow in @BeforeClass setup
5. Meaningful method names and comments"""


def get_screenshot_tc_prompt() -> str:
    return f"""You are a Technical Test Lead.
Analyze this UI screenshot and generate test cases.

EVERY test case MUST start with these 7 login steps in CSV:
"Launch the following url https://t1-aeg-qa-a.eluxmkt.com/der/de/b2b/pre-login/","User should be able to launch the url","User is able to launch the url"
"Click on the Partner link from the portal","User should be able to click on the partner link","User is able to Click on the partner link"
"Verify that the user is redirected to Prelogin page","User should be able to view the Prelogin page","User is able to View the Prelogin page"
"Verify whether the user is able to see Login now and Contact us Buttons","User should be able to see Login now and Contact us Buttons","User is able to View the Login now and Contact us Buttons"
"Click on Login now from prelogin page","User should be able to click Login now button","User is able to Click on the login now button"
"Verify it is redirected to SAML login page","User should be able to view the SAML login page","User is able to view the SAML login page"
"Enter user credentials and login","User should be able to login and view Chiron home page","User is able to View the Chiron home page"

{INTERMEDIATE_STEPS_INSTRUCTION}

Generate 6-8 test cases based on UI elements in screenshot.

First show titles:
✅ Generated Test Cases:
1. Verify whether user is able to [title]
2. Verify whether user is not able to [title]

Then IMMEDIATELY provide CSV:
---CSV START---
Test Case Title,Steps to Reproduce,Expected Result,Actual Result
---CSV END---

CSV Rules — 4 COLUMNS:
Column 1 - Test Case Title: same title repeats for every step
Column 2 - Steps to Reproduce: ACTION (Launch/Click/Navigate/Enter/Verify)
Column 3 - Expected Result:
  POSITIVE = "User should be able to [action]"
  NEGATIVE = "User should not be able to [action]"
Column 4 - Actual Result:
  POSITIVE = "User is able to [action]"
  NEGATIVE = "User is not able to [action]"
- Include all 7 login steps for every TC
- Then 2-3 navigation steps
- Then final verification step
- DO NOT skip the CSV section!"""


def get_bdd_prompt(ac_text: str) -> str:
    return f"""Act as a BDD Expert. \
Convert to Gherkin format.

Acceptance Criteria:
{ac_text}

Feature: [Feature Name]

  Background:
    Given user navigates to pre-login URL
    And user clicks Partner link
    And user is redirected to Prelogin page
    And user clicks Login now
    And user logs in with Chiron credentials

  Scenario: [Positive scenario]
    Given [precondition]
    When [action]
    Then [expected result]

  Scenario: [Negative scenario]
    Given [precondition]
    When [invalid action]
    Then [error result]"""


def get_summary_prompt(
        test_cases: list, feature: str) -> str:
    tc_text = "\n".join([
        f"- {tc['Test Case Title']}"
        for tc in test_cases[:20]
    ])
    return f"""As a QA Test Lead, write a professional \
test summary report for:

Feature: {feature}
Total Test Cases: {len(test_cases)}

Test Cases:
{tc_text}

Write professional report with:
1. Executive Summary (2-3 lines)
2. Test Scope
3. Test Approach
4. Risk Areas Identified
5. Recommendation

Concise and suitable for QA Manager review."""


# ===============================
# 📊 PARSE & CSV
# ===============================
def parse_test_cases_to_list(raw_text: str) -> list:
    test_cases = []

    if "---CSV START---" in raw_text and "---CSV END---" in raw_text:
        csv_section = raw_text.split(
            "---CSV START---")[1].split("---CSV END---")[0].strip()
        lines = csv_section.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("test case title"):
                continue
            try:
                reader = csv.reader([line])
                for parts in reader:
                    if len(parts) >= 3:
                        title = parts[0].strip().strip('"')
                        step = parts[1].strip().strip('"')
                        expected = parts[2].strip().strip('"')
                        actual = parts[3].strip().strip('"') if len(parts) >= 4 else ""
                        if title and step and len(step) > 5:
                            test_cases.append({
                                "Test Case Title": title,
                                "Steps to Reproduce": step,
                                "Expected Result": expected,
                                "Actual Result": actual,
                                "Status": "Not Executed"
                            })
            except Exception:
                continue
        if test_cases:
            return test_cases

    lines = raw_text.split("\n")
    csv_started = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if (line.lower().startswith("test case title") and
                "steps to reproduce" in line.lower()):
            csv_started = True
            continue
        if csv_started:
            if line.startswith("---"):
                break
            try:
                reader = csv.reader([line])
                for parts in reader:
                    if len(parts) >= 3:
                        title = parts[0].strip().strip('"')
                        step = parts[1].strip().strip('"')
                        expected = parts[2].strip().strip('"')
                        actual = parts[3].strip().strip('"') if len(parts) >= 4 else ""
                        if title and step and len(step) > 5:
                            test_cases.append({
                                "Test Case Title": title,
                                "Steps to Reproduce": step,
                                "Expected Result": expected,
                                "Actual Result": actual,
                                "Status": "Not Executed"
                            })
            except Exception:
                continue
    if test_cases:
        return test_cases

    current_title = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if (("verify" in line.lower() or
             "validate" in line.lower() or
             "test case title:" in line.lower())
                and "|" not in line and len(line) > 15):
            current_title = line.replace(
                "Test Case Title:", "").strip("*#-_ :")
        elif "|" in line and current_title:
            parts = line.split("|", 1)
            if len(parts) == 2:
                step = parts[0].strip().strip("*- ")
                expected = parts[1].strip().strip("*- ")
                if step and expected and len(step) > 5:
                    test_cases.append({
                        "Test Case Title": current_title,
                        "Steps to Reproduce": step,
                        "Expected Result": expected,
                        "Actual Result": "",
                        "Status": "Not Executed"
                    })
    return test_cases


def generate_csv(test_cases: list) -> bytes:
    output = io.StringIO()
    fieldnames = [
        "Test Case Title",
        "Steps to Reproduce",
        "Expected Result",
        "Actual Result",
        "Status"
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    prev_title = ""
    for tc in test_cases:
        row = {
            "Test Case Title": tc["Test Case Title"]
            if tc["Test Case Title"] != prev_title else "",
            "Steps to Reproduce": tc.get(
                "Steps to Reproduce", ""),
            "Expected Result": tc.get("Expected Result", ""),
            "Actual Result": tc.get("Actual Result", ""),
            "Status": tc.get("Status", "Not Executed")
        }
        writer.writerow(row)
        prev_title = tc["Test Case Title"]
    return output.getvalue().encode("utf-8")


def extract_display_text(reply: str) -> str:
    display_text = reply
    if "---CSV START---" in reply:
        display_text = reply.split("---CSV START---")[0].strip()
    lines = display_text.split("\n")
    clean_lines = []
    for line in lines:
        lower = line.strip().lower()
        if (lower.startswith("test case title") and
                "steps to reproduce" in lower):
            break
        clean_lines.append(line)
    display_text = "\n".join(clean_lines).strip()
    return display_text


# ===============================
# 🖥️ SIDEBAR
# ===============================
st.sidebar.markdown("## 🧪 QA Assistant")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Upload Files")

uploaded_files = st.sidebar.file_uploader(
    "Screenshots, PDFs, Docs",
    type=["txt", "pdf", "png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)

current_file_names = (
    [f.name for f in uploaded_files] if uploaded_files else []
)
if current_file_names != st.session_state.prev_file_names:
    st.session_state.images = []
    st.session_state.file_text = ""
    st.session_state.prev_file_names = current_file_names
    if uploaded_files:
        all_text = ""
        for file in uploaded_files:
            if file.type == "text/plain":
                all_text += file.read().decode("utf-8") + "\n\n"
            elif file.type == "application/pdf":
                try:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            all_text += text + "\n\n"
                except Exception as e:
                    st.sidebar.error(f"❌ {file.name}: {e}")
            elif file.type in [
                "image/png", "image/jpg",
                "image/jpeg", "image/webp"
            ]:
                image = Image.open(file)
                st.sidebar.image(
                    image,
                    caption=file.name,
                    use_column_width=True
                )
                st.session_state.images.append(image)
        st.session_state.file_text = all_text.strip()
        if st.session_state.file_text:
            st.sidebar.success("✅ Document loaded!")
        if st.session_state.images:
            st.sidebar.success(
                f"✅ {len(st.session_state.images)} "
                f"screenshot(s) loaded!")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Quick Actions")
sidebar_action = None

if st.sidebar.button(
        "📋 Generate Test Cases",
        use_container_width=True):
    sidebar_action = "generate_tc"
if st.sidebar.button(
        "🤖 Generate Selenium Code",
        use_container_width=True):
    sidebar_action = "generate_selenium"
if st.sidebar.button(
        "🖼️ Analyze Screenshot",
        use_container_width=True):
    sidebar_action = "analyze_screenshot"
if st.sidebar.button(
        "📝 BDD Scenarios",
        use_container_width=True):
    sidebar_action = "generate_bdd"
if st.sidebar.button(
        "📄 Test Summary Report",
        use_container_width=True):
    sidebar_action = "summary_report"

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Downloads")
if st.session_state.last_test_cases:
    csv_data = generate_csv(st.session_state.last_test_cases)
    st.sidebar.download_button(
        label="📊 Download CSV",
        data=csv_data,
        file_name="test_cases.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.sidebar.info("Generate test cases to enable download")

st.sidebar.markdown("---")
if st.sidebar.button(
        "🗑️ Clear All",
        use_container_width=True):
    st.session_state.chat_history = []
    st.session_state.images = []
    st.session_state.file_text = ""
    st.session_state.prev_file_names = []
    st.session_state.last_test_cases = []
    st.session_state.last_ac = ""
    st.session_state.dashboard_data = None
    st.session_state.last_reply = ""
    st.session_state.last_feature = ""
    st.session_state.last_file_prefix = ""
    st.session_state.last_action = ""
    st.rerun()


# ===============================
# 🖥️ MAIN AREA
# ===============================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Paste Ticket Details")
    feature_name = st.text_input(
        "🎫 Ticket Name",
        placeholder=(
            "e.g. Alternate Product - "
            "PLP Reset Message in Header"
        )
    )
    ac_input = st.text_area(
        "📋 Paste Full Ticket Details Here "
        "(User Story + Requirements + AC)",
        height=300,
        placeholder="""Paste everything here together..."""
    )

with col2:
    st.markdown("### 🚀 Generate")
    st.markdown("<br>", unsafe_allow_html=True)
    btn_tc = st.button(
        "📋 Test Cases",
        use_container_width=True,
        type="primary"
    )
    st.markdown("<br>", unsafe_allow_html=True)
    btn_selenium = st.button(
        "🤖 Selenium Java",
        use_container_width=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    btn_bdd = st.button(
        "📝 BDD Scenarios",
        use_container_width=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    btn_screenshot = st.button(
        "🖼️ From Screenshot",
        use_container_width=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    btn_summary = st.button(
        "📄 Summary Report",
        use_container_width=True
    )

st.divider()
st.markdown("### 💬 Results")

# ✅ Always show chat history like ChatGPT
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ===============================
# 🎯 SHARED: Process and display test cases
# ===============================
def process_and_display_test_cases(reply: str, feature: str,
                                   ac_text: str,
                                   file_prefix: str = ""):
    st.session_state.last_reply = reply
    st.session_state.last_feature = feature
    st.session_state.last_file_prefix = file_prefix

    display_text = extract_display_text(reply)
    st.markdown(display_text)

    parsed = parse_test_cases_to_list(reply)

    if parsed:
        st.session_state.last_test_cases = parsed
        st.session_state.last_ac = ac_text
        csv_data = generate_csv(parsed)
        unique_titles = list(dict.fromkeys(
            tc["Test Case Title"]
            for tc in parsed
            if tc.get("Test Case Title")
        ))
        st.success(
            f"✅ {len(unique_titles)} test cases generated! "
            f"Full steps + expected in CSV below."
        )
        fname = feature.replace(' ', '_')
        if file_prefix:
            fname = f"{fname}_{file_prefix}"
        st.download_button(
            label=(
                f"📊 Download Excel CSV "
                f"({len(unique_titles)} TCs, "
                f"{len(parsed)} rows)"
            ),
            data=csv_data,
            file_name=f"{fname}_test_cases.csv",
            mime="text/csv"
        )
        dash = compute_dashboard(parsed, ac_text)
        st.session_state.dashboard_data = (dash, feature)
        show_dashboard(dash, feature)
    else:
        st.warning(
            "⚠️ Could not parse structured test cases. "
            "Downloading raw output as fallback."
        )
        st.download_button(
            label="📊 Download Raw Output",
            data=reply.encode("utf-8"),
            file_name="test_cases_raw.txt",
            mime="text/plain"
        )

    return display_text


# ===============================
# 🎯 ACTION HANDLER
# ===============================
def handle_action(
        action_type: str,
        ac_text: str = "",
        feature: str = "Feature"):

    st.session_state.last_action = action_type

    if action_type == "generate_tc":
        if not ac_text.strip():
            st.warning(
                "⚠️ Please paste ticket details first!")
            return
        prompt = get_testcase_prompt(ac_text, feature)
        user_msg = (
            f"**📋 Generate Test Cases for:** {feature}"
            f"\n\n**Details:**\n{ac_text[:300]}..."
        )
        with st.chat_message("user"):
            st.markdown(user_msg)
        st.session_state.chat_history.append(
            {"role": "user", "content": user_msg})
        messages = [
            {"role": "system", "content": (
                "You are a QA expert. "
                "Show test case titles as numbered list. "
                "ALWAYS generate full CSV after titles. "
                "CSV starts with ---CSV START--- "
                "and ends with ---CSV END---. "
                "NEVER skip the CSV section. "
                "4 columns: Test Case Title, Steps to Reproduce, "
                "Expected Result, Actual Result. "
                "7 login steps first, then 2-3 nav steps, "
                "then final verification. "
                "Expected Result = User should be able to... "
                "Actual Result = User is able to..."
            )},
            {"role": "user", "content": prompt}
        ]
        with st.chat_message("assistant"):
            with st.spinner("🔍 Generating test cases..."):
                reply = call_groq(messages)
            display_text = process_and_display_test_cases(
                reply, feature, ac_text)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": display_text})

    elif action_type == "generate_selenium":
        if not ac_text.strip():
            st.warning(
                "⚠️ Please paste ticket details first!")
            return
        tc_context = "\n".join([
            f"{tc['Test Case Title']}"
            for tc in st.session_state.last_test_cases[:10]
        ]) if st.session_state.last_test_cases else ""
        prompt = get_selenium_prompt(ac_text, tc_context)
        user_msg = (
            f"**🤖 Generate Selenium Java Code for:** {feature}"
        )
        with st.chat_message("user"):
            st.markdown(user_msg)
        st.session_state.chat_history.append(
            {"role": "user", "content": user_msg})
        messages = [
            {"role": "system", "content": (
                "You are a Senior Selenium Automation Engineer. "
                "Generate clean production-ready Java code "
                "using TestNG and Page Object Model."
            )},
            {"role": "user", "content": prompt}
        ]
        with st.chat_message("assistant"):
            with st.spinner("⚙️ Generating Selenium Java..."):
                reply = call_groq(messages)
            st.markdown(reply)
            st.download_button(
                label="💾 Download .java file",
                data=reply.encode("utf-8"),
                file_name=(
                    f"{feature.replace(' ', '_')}_selenium.java"
                ),
                mime="text/plain"
            )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": reply})

    elif action_type == "analyze_screenshot":
        if not st.session_state.images:
            st.warning(
                "⚠️ Upload a screenshot from sidebar first!")
            return
        user_msg = (
            "**🖼️ Analyze Screenshot — Generate Test Cases**"
        )
        with st.chat_message("user"):
            st.markdown(user_msg)
        st.session_state.chat_history.append(
            {"role": "user", "content": user_msg})
        with st.chat_message("assistant"):
            with st.spinner(
                    "🔍 Step 1/2: Analyzing screenshot..."):
                vision_messages = [
                    {"role": "system", "content": (
                        "You are an expert UI analyst. "
                        "Describe every UI element you see "
                        "in the screenshot in detail."
                    )},
                    {"role": "user", "content": (
                        "Analyze this UI screenshot carefully "
                        "and list ALL UI elements you can see."
                    )}
                ]
                ui_description = call_groq(
                    vision_messages,
                    images=st.session_state.images
                )
            if ui_description.startswith("❌"):
                st.error(ui_description)
                st.session_state.chat_history.append(
                    {"role": "assistant",
                     "content": ui_description})
                return
            with st.spinner(
                    "📋 Step 2/2: Generating test cases..."):
                tc_prompt = get_testcase_prompt(
                    ui_description, feature)
                tc_messages = [
                    {"role": "system", "content": (
                        "You are a QA expert. "
                        "Show test case titles as numbered list. "
                        "ALWAYS generate full CSV after titles. "
                        "CSV starts with ---CSV START--- "
                        "and ends with ---CSV END---. "
                        "NEVER skip the CSV section. "
                        "4 columns: Test Case Title, "
                        "Steps to Reproduce, Expected Result, "
                        "Actual Result. "
                        "Expected Result = User should be able to... "
                        "Actual Result = User is able to... "
                        "NEVER repeat login steps after step 7."
                    )},
                    {"role": "user", "content": tc_prompt}
                ]
                try:
                    headers = {
                        "Authorization": f"Bearer {GROQ_KEY}",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "model": "llama-3.3-70b-versatile",
                        "messages": tc_messages,
                        "max_tokens": 7000,
                        "temperature": 0.3,
                    }
                    resp = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers, json=payload,
                        timeout=120,
                    )
                    resp.raise_for_status()
                    reply = resp.json(
                    )["choices"][0]["message"]["content"]
                except Exception as e:
                    reply = f"❌ Error: {str(e)}"
            screenshot_ac = (ac_text if ac_text.strip()
                             else ui_description)
            display_text = process_and_display_test_cases(
                reply,
                f"{feature} (Screenshot)",
                screenshot_ac,
                file_prefix="screenshot"
            )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": display_text})

    elif action_type == "generate_bdd":
        if not ac_text.strip():
            st.warning(
                "⚠️ Please paste ticket details first!")
            return
        prompt = get_bdd_prompt(ac_text)
        user_msg = f"**📝 BDD Scenarios for:** {feature}"
        with st.chat_message("user"):
            st.markdown(user_msg)
        st.session_state.chat_history.append(
            {"role": "user", "content": user_msg})
        messages = [
            {"role": "system", "content": (
                "You are a BDD expert. "
                "Generate clear Gherkin scenarios."
            )},
            {"role": "user", "content": prompt}
        ]
        with st.chat_message("assistant"):
            with st.spinner("📝 Generating BDD..."):
                reply = call_groq(messages)
            st.code(reply, language="gherkin")
            st.download_button(
                label="💾 Download .feature file",
                data=reply.encode("utf-8"),
                file_name=(
                    f"{feature.replace(' ', '_')}.feature"
                ),
                mime="text/plain"
            )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": reply})

    elif action_type == "summary_report":
        if not st.session_state.last_test_cases:
            st.warning(
                "⚠️ Generate test cases first!")
            return
        prompt = get_summary_prompt(
            st.session_state.last_test_cases, feature)
        user_msg = (
            f"**📄 Test Summary Report for:** {feature}"
        )
        with st.chat_message("user"):
            st.markdown(user_msg)
        st.session_state.chat_history.append(
            {"role": "user", "content": user_msg})
        messages = [
            {"role": "system", "content": (
                "You are a QA Test Lead writing professional "
                "reports for QA Managers. "
                "Be concise and professional."
            )},
            {"role": "user", "content": prompt}
        ]
        with st.chat_message("assistant"):
            with st.spinner("📄 Generating report..."):
                reply = call_groq(messages)
            st.markdown(reply)
            st.download_button(
                label="💾 Download Report (.txt)",
                data=reply.encode("utf-8"),
                file_name=(
                    f"{feature.replace(' ', '_')}_report.txt"
                ),
                mime="text/plain"
            )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": reply})


# ===============================
# 🎯 TRIGGER BUTTONS
# ===============================
if btn_tc:
    handle_action(
        "generate_tc", ac_input, feature_name or "Feature")
elif btn_selenium:
    handle_action(
        "generate_selenium", ac_input, feature_name or "Feature")
elif btn_bdd:
    handle_action(
        "generate_bdd", ac_input, feature_name or "Feature")
elif btn_screenshot:
    handle_action(
        "analyze_screenshot", ac_input,
        feature_name or "Feature")
elif btn_summary:
    handle_action(
        "summary_report", ac_input, feature_name or "Feature")
elif sidebar_action == "generate_tc":
    handle_action(
        "generate_tc", ac_input, feature_name or "Feature")
elif sidebar_action == "generate_selenium":
    handle_action(
        "generate_selenium", ac_input, feature_name or "Feature")
elif sidebar_action == "analyze_screenshot":
    handle_action(
        "analyze_screenshot", ac_input,
        feature_name or "Feature")
elif sidebar_action == "generate_bdd":
    handle_action(
        "generate_bdd", ac_input, feature_name or "Feature")
elif sidebar_action == "summary_report":
    handle_action(
        "summary_report", ac_input, feature_name or "Feature")


# ===============================
# 🔄 PERSIST RESULTS ON RERUN
# ===============================
if (st.session_state.last_test_cases
        and st.session_state.last_reply
        and not btn_tc
        and not btn_screenshot
        and sidebar_action not in ["generate_tc", "analyze_screenshot"]
        and st.session_state.last_action in [
            "generate_tc", "analyze_screenshot"]):

    feature = st.session_state.last_feature
    parsed = st.session_state.last_test_cases
    ac_text = st.session_state.last_ac

    display_text = extract_display_text(
        st.session_state.last_reply)
    if display_text:
        with st.chat_message("assistant"):
            st.markdown(display_text)
            csv_data = generate_csv(parsed)
            unique_titles = list(dict.fromkeys(
                tc["Test Case Title"]
                for tc in parsed
                if tc.get("Test Case Title")
            ))
            st.success(
                f"✅ {len(unique_titles)} test cases generated! "
                f"Full steps + expected in CSV below."
            )
            fname = feature.replace(' ', '_')
            if st.session_state.last_file_prefix:
                fname = (f"{fname}_"
                         f"{st.session_state.last_file_prefix}")
            st.download_button(
                label=(
                    f"📊 Download Excel CSV "
                    f"({len(unique_titles)} TCs, "
                    f"{len(parsed)} rows)"
                ),
                data=csv_data,
                file_name=f"{fname}_test_cases.csv",
                mime="text/csv",
                key="persist_download"
            )
            dash = compute_dashboard(parsed, ac_text)
            show_dashboard(dash, feature)


# ===============================
# 💬 FREE CHAT
# ===============================
st.divider()
user_prompt = st.chat_input(
    "Ask anything about QA, testing, automation...")

if user_prompt:
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.chat_history.append(
        {"role": "user", "content": user_prompt})

    system_msg = (
        "You are an expert QA Engineer and Technical Test Lead. "
        "Help with test cases, Selenium Java, bug reports, "
        "and QA best practices. "
        "Expected Result = User should be able to... "
        "Actual Result = User is able to..."
    )
    if st.session_state.file_text:
        system_msg += (
            f"\n\nDoc context:\n"
            f"{st.session_state.file_text[:4000]}"
        )
    if st.session_state.last_ac:
        system_msg += (
            f"\n\nPrevious AC:\n{st.session_state.last_ac}"
        )

    api_messages = [{"role": "system", "content": system_msg}]
    for msg in st.session_state.chat_history[-10:]:
        api_messages.append(msg)

    use_image = (
        st.session_state.images and
        any(w in user_prompt.lower() for w in [
            "image", "screenshot", "screen",
            "this", "describe", "analyze"
        ])
    )
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = call_groq(
                api_messages,
                images=(
                    st.session_state.images
                    if use_image else None
                )
            )
        st.markdown(reply)
    st.session_state.chat_history.append(
        {"role": "assistant", "content": reply})
