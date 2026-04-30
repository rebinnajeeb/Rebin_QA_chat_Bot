from dotenv import load_dotenv
import streamlit as st
import PyPDF2
from PIL import Image
import anthropic
import base64
import os
import io
import csv
import zipfile

# ---------- LOAD ENV ----------
load_dotenv()
CLAUDE_KEY = os.getenv("CLAUDE_API_KEY")

if not CLAUDE_KEY:
    st.error("❌ CLAUDE_API_KEY missing. Check .env file")
    st.stop()

client = anthropic.Anthropic(api_key=CLAUDE_KEY)

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
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🧪 QA Test Assistant</h1>
    <p>AI-Powered Test Case Generator | Selenium Code | Coverage Analyzer</p>
</div>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
defaults = {
    "chat_history": [],
    "images": [],
    "file_text": "",
    "prev_file_names": [],
    "last_test_cases": [],
    "last_ac": "",
    "dashboard_data": None,
    "last_reply": "",
    "last_feature": "",
    "last_file_prefix": "",
    "last_action": "",
    "dl_csv_data": None,
    "dl_csv_filename": "",
    "dl_csv_label": "",
    "dl_selenium_data": None,
    "dl_selenium_filename": "",
    "dl_bdd_data": None,
    "dl_bdd_filename": "",
    "dl_report_data": None,
    "dl_report_filename": "",
    "rendered_blocks": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ===============================
# 🔧 HELPERS
# ===============================
def image_to_base64(image: Image.Image) -> str:
    image = image.convert("RGB")
    image.thumbnail((1024, 1024))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def sanitize_messages(messages: list) -> list:
    """
    Claude API requires strictly alternating user/assistant messages.
    This function ensures no two consecutive same-role messages.
    """
    if not messages:
        return []
    sanitized = [messages[0]]
    for msg in messages[1:]:
        if msg["role"] != sanitized[-1]["role"]:
            sanitized.append(msg)
        else:
            # Merge consecutive same-role messages
            sanitized[-1] = {
                "role": msg["role"],
                "content": sanitized[-1]["content"] + "\n" + msg["content"]
            }
    return sanitized


def call_claude(messages: list, system: str = "", images: list = None) -> str:
    try:
        api_messages = []

        if images:
            for i, msg in enumerate(messages):
                if i == len(messages) - 1 and msg["role"] == "user":
                    content = []
                    for img in images:
                        b64 = image_to_base64(img)
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        })
                    content.append({"type": "text", "text": msg["content"]})
                    api_messages.append({"role": "user", "content": content})
                else:
                    api_messages.append({"role": msg["role"], "content": msg["content"]})
        else:
            api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

        # Sanitize to ensure alternating roles
        api_messages = sanitize_messages(api_messages)

        # Must start with user
        if api_messages and api_messages[0]["role"] != "user":
            api_messages = api_messages[1:]

        if not api_messages:
            return "❌ No valid messages to send."

        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=16000,
            system=system if system else "You are a helpful assistant.",
            messages=api_messages,
        )
        return response.content[0].text

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ===============================
# 🔑 NEGATIVE TITLE CHECK
# ===============================
def is_negative_title(title: str) -> bool:
    return "not able" in title.lower()


# ===============================
# 📦 ZIP HELPER
# ===============================
def create_zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, content in files.items():
            zf.writestr(filename, content)
    return buf.getvalue()


def parse_multi_file_response(reply: str) -> dict:
    files = {}
    parts = reply.split("===FILE:")
    for part in parts[1:]:
        lines = part.strip().split("\n")
        filename = lines[0].replace("===", "").strip()
        content = "\n".join(lines[1:]).strip()
        if filename and content:
            files[filename] = content
    return files


# ===============================
# 📊 DASHBOARD
# ===============================
def compute_dashboard(test_cases: list, ac_text: str) -> dict:
    unique_titles = list(dict.fromkeys(
        tc["Test Case Title"] for tc in test_cases
        if tc.get("Test Case Title")
    ))
    negative = sum(1 for t in unique_titles if is_negative_title(t))
    positive = len(unique_titles) - negative
    high = sum(
        1 for t in unique_titles
        if any(word in t.lower() for word in [
            "login", "security", "payment", "critical",
            "error", "not able", "invalid", "fail", "unable"
        ])
    )
    med = max(0, len(unique_titles) - high)
    ac_lines = [
        l.strip() for l in ac_text.split("\n")
        if l.strip() and len(l.strip()) > 10
    ]
    covered = 0
    for ac_line in ac_lines:
        keywords = [w for w in ac_line.lower().split() if len(w) > 4][:3]
        for tc in test_cases:
            tc_text = (
                tc.get("Steps to Reproduce", "") +
                tc.get("Test Case Title", "")
            ).lower()
            if any(kw in tc_text for kw in keywords):
                covered += 1
                break
    coverage_pct = int((covered / max(len(ac_lines), 1)) * 100)
    return {
        "total": len(unique_titles),
        "positive": positive,
        "negative": negative,
        "unique_tcs": len(unique_titles),
        "high": high,
        "med": med,
        "coverage_pct": coverage_pct,
        "unique_titles": unique_titles,
    }


def render_dashboard(data: dict, feature: str):
    st.markdown("---")
    st.markdown(f"### 📊 Live Dashboard — *{feature}*")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#185FA5">{data["unique_tcs"]}</div><div class="metric-lbl">Total Test Cases</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#3B6D11">{data["positive"]}</div><div class="metric-lbl">Positive Test Cases</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#A32D2D">{data["negative"]}</div><div class="metric-lbl">Negative Test Cases</div></div>', unsafe_allow_html=True)
    with c4:
        color = "#3B6D11" if data["coverage_pct"] >= 80 else "#854F0B" if data["coverage_pct"] >= 50 else "#A32D2D"
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:{color}">{data["coverage_pct"]}%</div><div class="metric-lbl">AC Coverage</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🎯 Coverage Breakdown")
    pos_pct = int((data["positive"] / max(data["total"], 1)) * 100)
    neg_pct = int((data["negative"] / max(data["total"], 1)) * 100)
    st.markdown(f"**Positive cases** — {pos_pct}%")
    st.progress(pos_pct / 100)
    st.markdown(f"**Negative cases** — {neg_pct}%")
    st.progress(neg_pct / 100)
    st.markdown(f"**AC Coverage** — {data['coverage_pct']}%")
    st.progress(data["coverage_pct"] / 100)
    st.markdown("#### 🏷️ Priority Distribution")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f'<span class="badge-high">🔴 High: {data["high"]}</span>', unsafe_allow_html=True)
    with p2:
        st.markdown(f'<span class="badge-med">🟡 Medium: {data["med"]}</span>', unsafe_allow_html=True)
    st.markdown("#### 📋 Test Cases Generated")
    for i, title in enumerate(data["unique_titles"], 1):
        badge = "neg" if is_negative_title(title) else "pos"
        label = "Negative" if badge == "neg" else "Positive"
        st.markdown(f'{i}. <span class="badge-{badge}">{label}</span> {title}', unsafe_allow_html=True)


# ===============================
# 📋 PROMPTS
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


def get_testcase_prompt(ac_text: str, feature_name: str = "Feature") -> str:
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

Generate as many test cases as needed to fully cover ALL acceptance criteria.
Mix positive and negative naturally based on AC — do NOT force equal numbers.

STRICT TITLE FORMAT:
POSITIVE = "Verify whether user is able to [action]"
NEGATIVE = "Verify whether user is not able to [action]"

First show titles only:
✅ Generated Test Cases:
1. Verify whether user is able to [title]
2. Verify whether user is not able to [title]

Then IMMEDIATELY provide full CSV:
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
- Complete ALL test cases — do NOT stop in the middle!"""


def get_selenium_prompt(ac_text: str, tc_text: str = "", feature: str = "Feature") -> str:
    return f"""Act as a Senior Selenium Automation Engineer.

Generate COMPLETE Selenium Java automation code for:
Feature: {feature}

Acceptance Criteria:
{ac_text}

{f"Test Cases:{chr(10)}{tc_text}" if tc_text else ""}

Generate ALL 4 files clearly separated:

===FILE: PageObject.java===
(Page Object Model — all @FindBy locators and action methods)

===FILE: TestNGTest.java===
(TestNG test class — @BeforeClass with login, @Test methods, @AfterClass)

===FILE: testng.xml===
(TestNG suite XML config)

===FILE: pom.xml===
(Maven pom with Selenium, TestNG, WebDriverManager dependencies)

Rules:
- Login URL: https://t1-aeg-qa-a.eluxmkt.com/der/de/b2b/pre-login/
- Use WebDriverManager for driver setup
- Use TestNG Assert for assertions
- Include both positive and negative test methods
- Add meaningful comments
- Make code complete and production ready"""


def get_bdd_prompt(ac_text: str, feature: str = "Feature") -> str:
    fname = feature.replace(" ", "")
    return f"""Act as a Senior BDD Automation Engineer.

Generate COMPLETE Cucumber BDD automation code for:
Feature: {feature}

Acceptance Criteria:
{ac_text}

Generate ALL 4 files clearly separated:

===FILE: {fname}.feature===
(Gherkin feature file — Background with login steps, positive and negative Scenarios)

===FILE: StepDefinitions.java===
(Java step definitions — EVERY Given/When/Then must have matching method)

===FILE: PageObject.java===
(Page Object Model — @FindBy locators and action methods)

===FILE: pom.xml===
(Maven pom with Selenium, Cucumber, TestNG, WebDriverManager dependencies)

Rules:
- Login URL: https://t1-aeg-qa-a.eluxmkt.com/der/de/b2b/pre-login/
- Background must include full login flow
- Every feature file step MUST have matching StepDefinitions method
- Use @FindBy in PageObject
- Add meaningful comments
- Make code complete and production ready"""


def get_summary_prompt(test_cases: list, feature: str) -> str:
    tc_text = "\n".join([f"- {tc['Test Case Title']}" for tc in test_cases[:20]])
    return f"""As a QA Test Lead, write a professional test summary report for:

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
        csv_section = (
            raw_text.split("---CSV START---")[1]
            .split("---CSV END---")[0]
            .strip()
        )
        lines = csv_section.split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.lower().startswith("test case title"):
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
                                "Status": "Not Executed",
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
        if (
            line.lower().startswith("test case title")
            and "steps to reproduce" in line.lower()
        ):
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
                                "Status": "Not Executed",
                            })
            except Exception:
                continue
    return test_cases


def generate_csv(test_cases: list) -> bytes:
    output = io.StringIO()
    fieldnames = [
        "Test Case Title", "Steps to Reproduce",
        "Expected Result", "Actual Result", "Status",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    prev_title = ""
    for tc in test_cases:
        row = {
            "Test Case Title": tc["Test Case Title"] if tc["Test Case Title"] != prev_title else "",
            "Steps to Reproduce": tc.get("Steps to Reproduce", ""),
            "Expected Result": tc.get("Expected Result", ""),
            "Actual Result": tc.get("Actual Result", ""),
            "Status": tc.get("Status", "Not Executed"),
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
        if lower.startswith("test case title") and "steps to reproduce" in lower:
            break
        clean_lines.append(line)
    return "\n".join(clean_lines).strip()


# ===============================
# 🗂️ RENDERED BLOCKS
# ===============================
def push_block(block: dict):
    st.session_state.rendered_blocks.append(block)


def render_block(block: dict, idx: int):
    btype = block["type"]

    if btype == "chat":
        with st.chat_message(block["role"]):
            st.markdown(block["content"])

    elif btype == "tc_result":
        with st.chat_message("assistant"):
            st.markdown(block["display_text"])
            unique_titles = list(dict.fromkeys(
                tc["Test Case Title"]
                for tc in block["parsed"]
                if tc.get("Test Case Title")
            ))
            st.success(
                f"✅ {len(unique_titles)} test cases generated! "
                "Full steps + expected in CSV below."
            )
            st.download_button(
                label=f"📊 Download Excel CSV ({len(unique_titles)} TCs, {len(block['parsed'])} rows)",
                data=block["csv_bytes"],
                file_name=block["csv_filename"],
                mime="text/csv",
                key=f"dl_csv_{idx}",
            )
            render_dashboard(block["dashboard"], block["feature"])

    elif btype == "selenium_result":
        with st.chat_message("assistant"):
            files = block.get("files", {})
            if files:
                st.markdown("### 🤖 Selenium TestNG — Generated Files")
                st.info("📦 Download ZIP contains all 4 files ready for your Java project!")
                st.download_button(
                    label="📦 Download All Files (ZIP)",
                    data=block["zip_bytes"],
                    file_name=block["zip_filename"],
                    mime="application/zip",
                    key=f"dl_selenium_zip_{idx}",
                )
                st.markdown("---")
                for fname, content in files.items():
                    lang = "xml" if fname.endswith(".xml") else "java"
                    with st.expander(f"📄 {fname}"):
                        st.code(content, language=lang)
            else:
                st.markdown(block["content"])
                st.download_button(
                    label="💾 Download .java file",
                    data=block["java_bytes"],
                    file_name=block["java_filename"],
                    mime="text/plain",
                    key=f"dl_java_{idx}",
                )

    elif btype == "bdd_result":
        with st.chat_message("assistant"):
            files = block.get("files", {})
            if files:
                st.markdown("### 📝 Cucumber BDD — Generated Files")
                st.info("📦 Download ZIP contains all 4 files ready for your Java project!")
                st.download_button(
                    label="📦 Download All Files (ZIP)",
                    data=block["zip_bytes"],
                    file_name=block["zip_filename"],
                    mime="application/zip",
                    key=f"dl_bdd_zip_{idx}",
                )
                st.markdown("---")
                for fname, content in files.items():
                    lang = "gherkin" if fname.endswith(".feature") else "xml" if fname.endswith(".xml") else "java"
                    with st.expander(f"📄 {fname}"):
                        st.code(content, language=lang)
            else:
                st.code(block["content"], language="gherkin")
                st.download_button(
                    label="💾 Download .feature file",
                    data=block["feature_bytes"],
                    file_name=block["feature_filename"],
                    mime="text/plain",
                    key=f"dl_bdd_{idx}",
                )

    elif btype == "report_result":
        with st.chat_message("assistant"):
            st.markdown(block["content"])
            st.download_button(
                label="💾 Download Report (.txt)",
                data=block["report_bytes"],
                file_name=block["report_filename"],
                mime="text/plain",
                key=f"dl_report_{idx}",
            )

    elif btype == "warning":
        st.warning(block["content"])

    elif btype == "error":
        st.error(block["content"])


def render_all_blocks():
    for idx, block in enumerate(st.session_state.rendered_blocks):
        render_block(block, idx)


# ===============================
# 🖥️ SIDEBAR
# ===============================
st.sidebar.markdown("## 🧪 QA Assistant")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Upload Files")

uploaded_files = st.sidebar.file_uploader(
    "Screenshots, PDFs, Docs",
    type=["txt", "pdf", "png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

current_file_names = [f.name for f in uploaded_files] if uploaded_files else []
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
            elif file.type in ["image/png", "image/jpg", "image/jpeg", "image/webp"]:
                image = Image.open(file)
                st.sidebar.image(image, caption=file.name, use_container_width=True)
                st.session_state.images.append(image)
        st.session_state.file_text = all_text.strip()
        if st.session_state.file_text:
            st.sidebar.success("✅ Document loaded!")
        if st.session_state.images:
            st.sidebar.success(f"✅ {len(st.session_state.images)} screenshot(s) loaded!")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Quick Actions")
sidebar_action = None

if st.sidebar.button("📋 Generate Test Cases", use_container_width=True):
    sidebar_action = "generate_tc"
if st.sidebar.button("🤖 Generate Selenium Code", use_container_width=True):
    sidebar_action = "generate_selenium"
if st.sidebar.button("🖼️ Analyze Screenshot", use_container_width=True):
    sidebar_action = "analyze_screenshot"
if st.sidebar.button("📝 BDD Scenarios", use_container_width=True):
    sidebar_action = "generate_bdd"
if st.sidebar.button("📄 Test Summary Report", use_container_width=True):
    sidebar_action = "summary_report"

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Downloads")

if st.session_state.dl_csv_data:
    st.sidebar.download_button(
        label=st.session_state.dl_csv_label or "📊 Download CSV",
        data=st.session_state.dl_csv_data,
        file_name=st.session_state.dl_csv_filename or "test_cases.csv",
        mime="text/csv",
        use_container_width=True,
        key="sidebar_dl_csv",
    )
else:
    st.sidebar.info("Generate test cases to enable download")

if st.session_state.dl_selenium_data:
    st.sidebar.download_button(
        label="📦 Download Selenium ZIP",
        data=st.session_state.dl_selenium_data,
        file_name=st.session_state.dl_selenium_filename or "selenium.zip",
        mime="application/zip",
        use_container_width=True,
        key="sidebar_dl_selenium",
    )

if st.session_state.dl_bdd_data:
    st.sidebar.download_button(
        label="📦 Download BDD ZIP",
        data=st.session_state.dl_bdd_data,
        file_name=st.session_state.dl_bdd_filename or "bdd.zip",
        mime="application/zip",
        use_container_width=True,
        key="sidebar_dl_bdd",
    )

if st.session_state.dl_report_data:
    st.sidebar.download_button(
        label="💾 Download Report .txt",
        data=st.session_state.dl_report_data,
        file_name=st.session_state.dl_report_filename or "report.txt",
        mime="text/plain",
        use_container_width=True,
        key="sidebar_dl_report",
    )

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear All", use_container_width=True):
    for k, v in defaults.items():
        st.session_state[k] = v if not callable(v) else v()
    st.rerun()


# ===============================
# 🖥️ MAIN AREA
# ===============================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Paste Ticket Details")
    feature_name = st.text_input(
        "🎫 Ticket Name",
        placeholder="e.g. Alternate Product - PLP Reset Message in Header",
    )
    ac_input = st.text_area(
        "📋 Paste Full Ticket Details Here (User Story + Requirements + AC)",
        height=300,
        placeholder="Paste everything here together...",
    )

with col2:
    st.markdown("### 🚀 Generate")
    st.markdown("<br>", unsafe_allow_html=True)
    btn_tc = st.button("📋 Test Cases", use_container_width=True, type="primary")
    st.markdown("<br>", unsafe_allow_html=True)
    btn_selenium = st.button("🤖 Selenium Java", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    btn_bdd = st.button("📝 BDD Scenarios", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    btn_screenshot = st.button("🖼️ From Screenshot", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    btn_summary = st.button("📄 Summary Report", use_container_width=True)

st.divider()
st.markdown("### 💬 Results")
render_all_blocks()


# ===============================
# 🎯 ACTION HANDLERS
# ===============================
def handle_generate_tc(ac_text: str, feature: str):
    if not ac_text.strip():
        st.warning("⚠️ Please paste ticket details first!")
        return

    prompt = get_testcase_prompt(ac_text, feature)
    user_msg = f"**📋 Generate Test Cases for:** {feature}\n\n**Details:**\n{ac_text[:300]}..."
    push_block({"type": "chat", "role": "user", "content": user_msg})
    st.session_state.chat_history.append({"role": "user", "content": user_msg})

    with st.spinner("🔍 Generating test cases..."):
        reply = call_claude(
            messages=[{"role": "user", "content": prompt}],
            system=(
                "You are a QA expert. "
                "STRICT RULE: Every Test Case Title MUST start with "
                "'Verify whether user is able to' OR 'Verify whether user is not able to'. "
                "Show titles as numbered list first, then ALWAYS generate full CSV. "
                "CSV starts with ---CSV START--- and ends with ---CSV END---. "
                "NEVER skip CSV. Complete ALL test cases without stopping."
            ),
        )

    display_text = extract_display_text(reply)
    parsed = parse_test_cases_to_list(reply)
    st.session_state.last_reply = reply
    st.session_state.last_feature = feature
    st.session_state.last_ac = ac_text

    if parsed:
        st.session_state.last_test_cases = parsed
        csv_bytes = generate_csv(parsed)
        unique_titles = list(dict.fromkeys(
            tc["Test Case Title"] for tc in parsed if tc.get("Test Case Title")
        ))
        fname = feature.replace(" ", "_")
        csv_filename = f"{fname}_test_cases.csv"
        dash = compute_dashboard(parsed, ac_text)
        st.session_state.dl_csv_data = csv_bytes
        st.session_state.dl_csv_filename = csv_filename
        st.session_state.dl_csv_label = f"📊 Download CSV ({len(unique_titles)} TCs)"
        push_block({
            "type": "tc_result",
            "display_text": display_text,
            "parsed": parsed,
            "csv_bytes": csv_bytes,
            "csv_filename": csv_filename,
            "dashboard": dash,
            "feature": feature,
        })
    else:
        push_block({"type": "chat", "role": "assistant", "content": display_text})
        push_block({"type": "warning", "content": "⚠️ Could not parse structured test cases. Raw output shown above."})

    st.session_state.chat_history.append({"role": "assistant", "content": display_text})
    st.rerun()


def handle_generate_selenium(ac_text: str, feature: str):
    if not ac_text.strip():
        st.warning("⚠️ Please paste ticket details first!")
        return

    tc_context = "\n".join([
        tc["Test Case Title"]
        for tc in st.session_state.last_test_cases[:10]
    ]) if st.session_state.last_test_cases else ""

    prompt = get_selenium_prompt(ac_text, tc_context, feature)
    user_msg = f"**🤖 Generate Selenium TestNG Code for:** {feature}"
    push_block({"type": "chat", "role": "user", "content": user_msg})
    st.session_state.chat_history.append({"role": "user", "content": user_msg})

    with st.spinner("⚙️ Generating Selenium TestNG files..."):
        reply = call_claude(
            messages=[{"role": "user", "content": prompt}],
            system=(
                "You are a Senior Selenium Automation Engineer. "
                "Generate ALL 4 files separated by ===FILE: filename=== markers. "
                "Files: PageObject.java, TestNGTest.java, testng.xml, pom.xml. "
                "Make code complete and production ready."
            ),
        )

    files = parse_multi_file_response(reply)
    fname = feature.replace(" ", "_")

    if files:
        zip_bytes = create_zip(files)
        zip_filename = f"{fname}_selenium_testng.zip"
        st.session_state.dl_selenium_data = zip_bytes
        st.session_state.dl_selenium_filename = zip_filename
        push_block({
            "type": "selenium_result",
            "files": files,
            "zip_bytes": zip_bytes,
            "zip_filename": zip_filename,
            "content": reply,
            "java_bytes": reply.encode("utf-8"),
            "java_filename": f"{fname}_selenium.java",
        })
    else:
        java_bytes = reply.encode("utf-8")
        java_filename = f"{fname}_selenium.java"
        st.session_state.dl_selenium_data = java_bytes
        st.session_state.dl_selenium_filename = java_filename
        push_block({
            "type": "selenium_result",
            "files": {},
            "content": reply,
            "java_bytes": java_bytes,
            "java_filename": java_filename,
            "zip_bytes": b"",
            "zip_filename": "",
        })

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.rerun()


def handle_analyze_screenshot(ac_text: str, feature: str):
    if not st.session_state.images:
        st.warning("⚠️ Upload a screenshot from sidebar first!")
        return

    user_msg = "**🖼️ Analyze Screenshot — Generate Test Cases**"
    push_block({"type": "chat", "role": "user", "content": user_msg})
    st.session_state.chat_history.append({"role": "user", "content": user_msg})

    with st.spinner("🔍 Step 1/2: Analyzing screenshot..."):
        ui_description = call_claude(
            messages=[{"role": "user", "content": "Analyze this UI screenshot carefully and describe every UI element, button, field, text, and section you can see in detail."}],
            system="You are an expert UI analyst. Describe every UI element you see in the screenshot in complete detail.",
            images=st.session_state.images,
        )

    if ui_description.startswith("❌"):
        push_block({"type": "error", "content": ui_description})
        st.rerun()
        return

    with st.spinner("📋 Step 2/2: Generating test cases..."):
        tc_prompt = get_testcase_prompt(ui_description, feature)
        reply = call_claude(
            messages=[{"role": "user", "content": tc_prompt}],
            system=(
                "You are a QA expert. "
                "STRICT RULE: Every Test Case Title MUST start with "
                "'Verify whether user is able to' OR 'Verify whether user is not able to'. "
                "Show titles as numbered list first, then ALWAYS generate full CSV. "
                "CSV starts with ---CSV START--- and ends with ---CSV END---. "
                "NEVER repeat login steps after step 7. "
                "Complete ALL test cases without stopping."
            ),
        )

    screenshot_ac = ac_text if ac_text.strip() else ui_description
    display_text = extract_display_text(reply)
    parsed = parse_test_cases_to_list(reply)
    st.session_state.last_reply = reply
    st.session_state.last_feature = f"{feature} (Screenshot)"
    st.session_state.last_ac = screenshot_ac

    if parsed:
        st.session_state.last_test_cases = parsed
        csv_bytes = generate_csv(parsed)
        unique_titles = list(dict.fromkeys(
            tc["Test Case Title"] for tc in parsed if tc.get("Test Case Title")
        ))
        fname = f"{feature.replace(' ', '_')}_screenshot"
        csv_filename = f"{fname}_test_cases.csv"
        dash = compute_dashboard(parsed, screenshot_ac)
        st.session_state.dl_csv_data = csv_bytes
        st.session_state.dl_csv_filename = csv_filename
        st.session_state.dl_csv_label = f"📊 Download CSV ({len(unique_titles)} TCs)"
        push_block({
            "type": "tc_result",
            "display_text": display_text,
            "parsed": parsed,
            "csv_bytes": csv_bytes,
            "csv_filename": csv_filename,
            "dashboard": dash,
            "feature": f"{feature} (Screenshot)",
        })
    else:
        push_block({"type": "chat", "role": "assistant", "content": display_text})
        push_block({"type": "warning", "content": "⚠️ Could not parse structured test cases. Raw output shown above."})

    st.session_state.chat_history.append({"role": "assistant", "content": display_text})
    st.rerun()


def handle_generate_bdd(ac_text: str, feature: str):
    if not ac_text.strip():
        st.warning("⚠️ Please paste ticket details first!")
        return

    prompt = get_bdd_prompt(ac_text, feature)
    user_msg = f"**📝 Generate BDD Cucumber Code for:** {feature}"
    push_block({"type": "chat", "role": "user", "content": user_msg})
    st.session_state.chat_history.append({"role": "user", "content": user_msg})

    with st.spinner("📝 Generating BDD Cucumber files..."):
        reply = call_claude(
            messages=[{"role": "user", "content": prompt}],
            system=(
                "You are a Senior BDD Automation Engineer. "
                "Generate ALL 4 files separated by ===FILE: filename=== markers. "
                "Files: .feature file, StepDefinitions.java, PageObject.java, pom.xml. "
                "Every step in feature file MUST have matching method in StepDefinitions.java. "
                "Make code complete and production ready."
            ),
        )

    files = parse_multi_file_response(reply)
    fname = feature.replace(" ", "_")

    if files:
        zip_bytes = create_zip(files)
        zip_filename = f"{fname}_bdd_cucumber.zip"
        st.session_state.dl_bdd_data = zip_bytes
        st.session_state.dl_bdd_filename = zip_filename
        push_block({
            "type": "bdd_result",
            "files": files,
            "zip_bytes": zip_bytes,
            "zip_filename": zip_filename,
            "content": reply,
            "feature_bytes": reply.encode("utf-8"),
            "feature_filename": f"{fname}.feature",
        })
    else:
        feature_bytes = reply.encode("utf-8")
        feature_filename = f"{fname}.feature"
        st.session_state.dl_bdd_data = feature_bytes
        st.session_state.dl_bdd_filename = feature_filename
        push_block({
            "type": "bdd_result",
            "files": {},
            "content": reply,
            "feature_bytes": feature_bytes,
            "feature_filename": feature_filename,
            "zip_bytes": b"",
            "zip_filename": "",
        })

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.rerun()


def handle_summary_report(ac_text: str, feature: str):
    if not st.session_state.last_test_cases:
        st.warning("⚠️ Generate test cases first!")
        return

    prompt = get_summary_prompt(st.session_state.last_test_cases, feature)
    user_msg = f"**📄 Test Summary Report for:** {feature}"
    push_block({"type": "chat", "role": "user", "content": user_msg})
    st.session_state.chat_history.append({"role": "user", "content": user_msg})

    with st.spinner("📄 Generating report..."):
        reply = call_claude(
            messages=[{"role": "user", "content": prompt}],
            system="You are a QA Test Lead writing professional reports for QA Managers. Be concise and professional.",
        )

    report_bytes = reply.encode("utf-8")
    report_filename = f"{feature.replace(' ', '_')}_report.txt"
    st.session_state.dl_report_data = report_bytes
    st.session_state.dl_report_filename = report_filename
    push_block({
        "type": "report_result",
        "content": reply,
        "report_bytes": report_bytes,
        "report_filename": report_filename,
    })
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.rerun()


# ===============================
# 🎯 TRIGGER BUTTONS
# ===============================
active_feature = feature_name or "Feature"

if btn_tc:
    handle_generate_tc(ac_input, active_feature)
elif btn_selenium:
    handle_generate_selenium(ac_input, active_feature)
elif btn_bdd:
    handle_generate_bdd(ac_input, active_feature)
elif btn_screenshot:
    handle_analyze_screenshot(ac_input, active_feature)
elif btn_summary:
    handle_summary_report(ac_input, active_feature)
elif sidebar_action == "generate_tc":
    handle_generate_tc(ac_input, active_feature)
elif sidebar_action == "generate_selenium":
    handle_generate_selenium(ac_input, active_feature)
elif sidebar_action == "analyze_screenshot":
    handle_analyze_screenshot(ac_input, active_feature)
elif sidebar_action == "generate_bdd":
    handle_generate_bdd(ac_input, active_feature)
elif sidebar_action == "summary_report":
    handle_summary_report(ac_input, active_feature)


# ===============================
# 💬 FREE CHAT
# ===============================
st.divider()
user_prompt = st.chat_input("Ask anything about QA, testing, automation...")

if user_prompt:
    push_block({"type": "chat", "role": "user", "content": user_prompt})
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    system_msg = (
        "You are an expert QA Engineer and Technical Test Lead. "
        "Help with test cases, Selenium Java, bug reports, and QA best practices. "
        "Expected Result = User should be able to... "
        "Actual Result = User is able to..."
    )
    if st.session_state.file_text:
        system_msg += f"\n\nDoc context:\n{st.session_state.file_text[:4000]}"
    if st.session_state.last_ac:
        system_msg += f"\n\nPrevious AC:\n{st.session_state.last_ac}"

    # Build clean alternating messages for Claude
    raw_history = st.session_state.chat_history[-10:]
    api_messages = sanitize_messages([
        {"role": m["role"], "content": m["content"]}
        for m in raw_history
    ])

    # Ensure last message is user
    if not api_messages or api_messages[-1]["role"] != "user":
        api_messages.append({"role": "user", "content": user_prompt})

    use_image = st.session_state.images and any(
        w in user_prompt.lower()
        for w in ["image", "screenshot", "screen", "this", "describe", "analyze"]
    )

    with st.spinner("Thinking..."):
        reply = call_claude(
            messages=api_messages,
            system=system_msg,
            images=(st.session_state.images if use_image else None),
        )

    push_block({"type": "chat", "role": "assistant", "content": reply})
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.rerun()
