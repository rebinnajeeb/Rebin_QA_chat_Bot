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

# ---------- PAGE CONFIG (must be first Streamlit call) ----------
st.set_page_config(
    page_title="QA Test Assistant",
    page_icon="🧪",
    layout="wide"
)

# ---------- LOAD ENV ----------
load_dotenv()
CLAUDE_KEY = os.getenv("CLAUDE_API_KEY")

if not CLAUDE_KEY:
    st.error("❌ CLAUDE_API_KEY missing. Check .env file")
    st.stop()

@st.cache_resource
def get_client():
    return anthropic.Anthropic(api_key=CLAUDE_KEY)

client = get_client()

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
    .badge-high { background:#ffe0e0; color:#c00; padding:2px 8px; border-radius:99px; font-size:11px; }
    .badge-med { background:#fff3cd; color:#856404; padding:2px 8px; border-radius:99px; font-size:11px; }
    .badge-pos { background:#d1f5ea; color:#0f6e56; padding:2px 8px; border-radius:99px; font-size:11px; }
    .badge-neg { background:#fde8e8; color:#a32d2d; padding:2px 8px; border-radius:99px; font-size:11px; }
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
    "selected_model": "claude-haiku-4-5",
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
# 💰 SMART TOKEN ALLOCATION
# (Anthropic safe limit = 21,333 — no streaming required)
# ===============================
TOKEN_BUDGETS = {
    "test_cases": 16000,    # CSV usually fits
    "selenium":   21000,    # MAX safe limit (no streaming needed)
    "bdd":        21000,    # MAX safe limit (no streaming needed)
    "screenshot": 16000,    # Vision + test cases
    "summary":     4000,    # Always small report
    "free_chat":   8000,    # Conversational
}


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
    if not messages:
        return []
    sanitized = [messages[0]]
    for msg in messages[1:]:
        if msg["role"] != sanitized[-1]["role"]:
            sanitized.append(msg)
        else:
            sanitized[-1] = {
                "role": msg["role"],
                "content": sanitized[-1]["content"] + "\n" + msg["content"]
            }
    return sanitized


def call_claude(messages: list, system: str = "", images: list = None, max_tokens: int = 16000) -> str:
    """
    Call Claude API with smart token allocation.
    max_tokens is LIMIT not CHARGE — only pay for what's actually generated.
    Max safe limit = 21,333 (above this, streaming is required).
    """
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

        api_messages = sanitize_messages(api_messages)

        if api_messages and api_messages[0]["role"] != "user":
            api_messages = api_messages[1:]

        if not api_messages:
            return "❌ No valid messages to send."

        response = client.messages.create(
            model=st.session_state.get("selected_model", "claude-haiku-4-5"),
            max_tokens=max_tokens,
            system=system if system else "You are a helpful assistant.",
            messages=api_messages,
        )
        return response.content[0].text

    except Exception as e:
        return f"❌ Error: {str(e)}"


def is_negative_title(title: str) -> bool:
    return "not able" in title.lower()


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


def build_full_tc_context(test_cases: list) -> str:
    """
    Build FULL test case context (titles + steps + expected)
    for passing to Selenium and BDD generators.
    """
    if not test_cases:
        return ""

    grouped = {}
    for tc in test_cases:
        title = tc.get("Test Case Title", "").strip()
        if not title:
            continue
        if title not in grouped:
            grouped[title] = []
        grouped[title].append({
            "step": tc.get("Steps to Reproduce", ""),
            "expected": tc.get("Expected Result", ""),
            "actual": tc.get("Actual Result", ""),
        })

    context = ""
    for idx, (title, steps) in enumerate(grouped.items(), 1):
        context += f"\n═══════════════════════════════════════\n"
        context += f"TEST CASE {idx}: {title}\n"
        context += f"═══════════════════════════════════════\n"
        for step_idx, s in enumerate(steps, 1):
            context += f"  Step {step_idx}: {s['step']}\n"
            context += f"    Expected: {s['expected']}\n"
    return context


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
# 🌐 CHIRON WEBSITE NAVIGATION KNOWLEDGE
# ===============================
WEBSITE_KNOWLEDGE = """
═══════════════════════════════════════════════════════
CHIRON (AEG PARTNER PORTAL) — WEBSITE NAVIGATION GUIDE
═══════════════════════════════════════════════════════

After step 7 (login complete), user lands on CHIRON HOME PAGE.

HOME PAGE LAYOUT:
- Top header: AEG Logo (center), Möbelland Hochtaunus GmbH (right), User name dropdown (right)
- Top menu (left): Household appliances | Marketing & Sales | Orders
- Top menu (right): Trainings | Tools & Services
- Top right icons: Notification bell 🔔 (with count) | Search 🔍 | Wishlist ❤️ (heart) | Basket 🛒 (with item count)
- Main banner: "WELCOME TO THE AEG PARTNER PORTAL"

═══════════════════════════════════════════════════════
NAVIGATION FLOWS — USE THESE FOR INTERMEDIATE STEPS
═══════════════════════════════════════════════════════

▶ TO REACH PLP (Product Listing Page):
  1. Click on "Household appliances" from top menu
  2. Mega menu opens with categories: Cooking | Sets | Cooling and freezing | Dishwashing | Kitchen Gadgets | Laundry Care | Housekeeping | Indoor climate | Accessories
  3. Click on any subcategory (use generic phrasing like "any PLP category")
  4. PLP loads with product cards

▶ TO REACH PDP (Product Details Page):
  1. From PLP, click on any product card or product image
  2. PDP opens showing product details, price, "Add to basket-B2B" button

▶ TO ADD PRODUCT TO BASKET:
  1. From PDP, click "Add to basket-B2B" red button
  2. Basket side panel opens on right
  3. Shows item count, product, price
  4. Click "View Basket-B2B" to go to checkout

▶ TO VIEW BASKET / GO TO CHECKOUT:
  1. Click on Basket icon 🛒 (top right) — OR click "View Basket-B2B" from side panel
  2. Checkout page opens with Delivery section, Products section, Price summary, Confirmation

▶ TO COMPLETE CHECKOUT:
  1. On Checkout page, fill "Purchase order ID"
  2. Verify "Selected store"
  3. Choose Delivery option (Store delivery / etc.)
  4. Choose Shipment type (Partial delivery / Complete order delivery)
  5. Set "Requested delivery date"
  6. Review Products list with quantities
  7. Review Price summary (Total amount, Discount, VAT, Total)
  8. Tick "Send order confirmation via email" + verify email
  9. Click submit/place order

▶ TO USE WISHLIST:
  1. Click Wishlist icon ❤️ (heart icon, top right)
  2. Wishlist page opens showing all saved favourites
  3. From PDP, click heart icon on product to add to wishlist

▶ TO USE SEARCH:
  1. Click Search icon 🔍 (top right)
  2. Search bar opens
  3. Type product name / PNC / Model ID
  4. Press Enter or click search

▶ TO USE FILTERS ON PLP:
  1. On PLP, left sidebar has filters (B2B-Stock, Test Range, EAN number, Device width, etc.)
  2. Click filter checkboxes
  3. Products auto-filter

▶ TO COMPARE PRODUCTS:
  1. On PLP, click "Compare Products" button on product card (max 3 products)
  2. Click compare button to view comparison

═══════════════════════════════════════════════════════
🔥 ALTERNATIVE PRODUCTS LOGIC — IMPORTANT (PDP)
═══════════════════════════════════════════════════════

CONDITION FOR BOTH BUTTONS:
Both buttons appear ONLY when product status is:
  • "Out of stock" OR
  • "Expected to be available from [date]"

(Both buttons NEVER appear for in-stock regular products)

DECISION LOGIC — Which button shows:

CASE A: Alternatives with >=50% similar facets EXIST
→ Button shown: "See Similar Products" (under Add to cart)
→ Click action: STAYS on same PDP
→ Behavior: Scrolls down → "Alternative Product Comparison Component" appears
→ Component shows:
  • Reference Product card
  • Best Match products (>=90% facet similarity)
  • Good Match products (>=75% facet similarity)
  • Each card: match %, price, Add to Cart, availability

CASE B: NO alternatives with >=50% similar facets
→ Button shown: "Explore Alternate" (also "Review Alternative Products")
→ Click action: REDIRECTS to NEW Alternative Product PLP
→ New PLP page shows:
  • "Start refining your result" filter message
  • Reference Product card FIRST with star icon + "Reference Product" label
  • Pre-selected facets (excluding Stock Level facet)
  • In-stock filter auto-enabled
  • Only 100% matching products displayed
  • If no matches: "Edit the filters to see suitable alternatives" message
  • URL pattern: /category/?viewAlternatives=PNC

KEY: Only ONE button shows at a time.

▶ TO ACCESS NOTIFICATIONS:
  1. Click Notification bell 🔔 (top right, with unread count)
  2. "My messages" page opens

▶ TO ACCESS MY ACCOUNT:
  1. Click on user name (top right)
  2. Dropdown shows: Mein Account | Admin | Logout

▶ TO REACH MARKETING & SALES:
  1. Click "Marketing & Sales" from top menu
  2. Mega menu shows: Campaigns | Promotions | Sales documents | Contact Us | News and Updates | Premier Line | Premier Partner | Core Range

▶ TO REACH ORDERS:
  1. Click "Orders" from top menu
  2. Mega menu shows: Job Status | Order search | Direct order | Contact us

═══════════════════════════════════════════════════════
RULES FOR INTERMEDIATE STEPS GENERATION
═══════════════════════════════════════════════════════

1. After step 7 (login), user is on CHIRON HOME PAGE
2. Use GENERIC phrasing for navigation (don't pick specific category):
   - "Navigate to any PLP category" (NOT "Cooking → Hobs")
   - "Select any product with status = Out of stock / Expected to be available"
3. Use REAL element names (Add to basket-B2B, See Similar Products, Explore Alternate, Reference Product)
4. NEVER repeat login steps after step 7
5. NEVER skip intermediate steps
6. Each step must be ACTIONABLE
"""


# ===============================
# 📋 PROMPTS
# ===============================
def get_testcase_prompt(ac_text: str, feature_name: str = "Feature") -> str:
    return f"""You are a Technical Test Lead for AEG Partner Portal (Chiron). Generate test cases for:

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

{WEBSITE_KNOWLEDGE}

═══════════════════════════════════════════════════════
🚨 STRICT 1:1 AC MAPPING — TEST CASE GENERATION
═══════════════════════════════════════════════════════

🔥 GOLDEN RULE: ONE AC POINT = ONE TEST CASE (MAX)
   - Count the AC bullet points
   - Generate AT MOST that many test cases
   - PREFER FEWER if multiple AC points test the same thing
   - NEVER generate MORE than AC points count

🚫 ZERO TOLERANCE FOR EXTRAS:
   ❌ NO "logically needed" basic test cases
   ❌ NO "nice to have" page load verification
   ❌ NO "just in case" UI element checks
   ❌ NO duplicate scenarios with slight variations
   ❌ NO test cases for things NOT in AC
   ❌ NO test cases for functionality from WEBSITE_KNOWLEDGE
      that isn't mentioned in AC

✅ ONLY GENERATE:
   ✓ One test case per AC bullet point
   ✓ Negative test ONLY if AC explicitly mentions a negative scenario
   ✓ Test cases that DIRECTLY verify a specific AC requirement

═══════════════════════════════════════════════════════
🚨 HOW TO COUNT AC POINTS — DO THIS FIRST
═══════════════════════════════════════════════════════

Before writing ANY test case:
1. READ the AC carefully
2. COUNT distinct testable points (each bullet, each "Then" statement)
3. WRITE that exact count in your output: "Detected X AC points"
4. Generate AT MOST X test cases
5. Each test case must DIRECTLY map to ONE AC point

EXAMPLE:
AC has these points:
   • First product = reference product (1 point)
   • Reference stays first regardless of filters (1 point)
   • "Reference Product" label + star icon (1 point)
   • All eligible facets auto-enabled (1 point)
   • Stock Level facet excluded (1 point)
   • In-stock filter auto-enabled (1 point)
   • Only 100% matching products (1 point)
   • Out of customer range → standard category page (1 point)

Total: 8 AC points → Generate AT MOST 8 test cases (could be 6-8)

═══════════════════════════════════════════════════════
🚨 USE WEBSITE_KNOWLEDGE ONLY FOR NAVIGATION
═══════════════════════════════════════════════════════

The WEBSITE_KNOWLEDGE above tells you HOW to navigate.
Use it ONLY for the steps to reach the page mentioned in AC.
Do NOT use it to invent extra test cases.

For example:
- AC mentions "alternative PLP" → use WEBSITE_KNOWLEDGE to know
  HOW to navigate there (login → mega menu → product → button)
- AC does NOT mention "Start refining your result" message →
  do NOT add a test for it, even though WEBSITE_KNOWLEDGE mentions it

═══════════════════════════════════════════════════════
🚨 CSV FORMAT RULES — STRICTLY FOLLOW THIS
═══════════════════════════════════════════════════════

1. CSV has EXACTLY 4 columns:
   Column 1: Test Case Title
   Column 2: Steps to Reproduce
   Column 3: Expected Result
   Column 4: Actual Result

2. NEVER put step numbers ("Step 1:", "Step 9:") in step text!
   ❌ WRONG: "Step 9: From the mega menu"
   ✅ CORRECT: "From the mega menu, click on any subcategory"

3. Title column = ONLY the test case title.
   Title is SAME for all rows of one test case.

4. Each row in CSV = ONE step of a test case.
   Title repeats for every step of the same test case.

5. EXAMPLE of CORRECT CSV (1 TC = multiple rows):
   "Verify whether user is able to see Reference Product","Launch URL...","Should launch","Able to launch"
   "Verify whether user is able to see Reference Product","Click Partner link","Should click","Able to click"
   "Verify whether user is able to see Reference Product","Click Household appliances","Should click","Able to click"
   (... title repeats for ALL steps of this TC ...)

═══════════════════════════════════════════════════════
🚨 STEPS TO REPRODUCE — DETAILED & GENERIC
═══════════════════════════════════════════════════════

6. Each step must be ACTIONABLE — tester can directly execute.

7. Use GENERIC phrasing for navigation (don't pick specific category):
   ❌ DON'T: "Click Cooking → Hobs → Gas hobs"
   ✅ DO: "Navigate to any PLP category from mega menu"
   ✅ DO: "Select any product with status = Out of stock / Expected to be available"

8. Write EXACT navigation flow — NEVER skip intermediate steps.

9. For ALTERNATIVE PRODUCTS test cases — use the EXACT logic from
   WEBSITE_KNOWLEDGE:
   - "See Similar Products" button → stays on PDP, comparison appears
   - "Explore Alternate" button → redirects to new Alternative PLP
   - Both buttons require: status = Out of stock / Expected to be available

═══════════════════════════════════════════════════════
TITLE FORMAT — STRICT
═══════════════════════════════════════════════════════

POSITIVE = "Verify whether user is able to [action]"
NEGATIVE = "Verify whether user is not able to [action]"

═══════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════

FIRST output the AC analysis:
📊 AC Analysis:
- Detected AC points: [count]
- Will generate: [count] test cases (1:1 mapping)

Then show titles only:
✅ Generated Test Cases:
1. Verify whether user is able to [title]
2. Verify whether user is not able to [title]

Then IMMEDIATELY provide full CSV between markers:
---CSV START---
Test Case Title,Steps to Reproduce,Expected Result,Actual Result
---CSV END---

CSV Rules:
Column 1 - Test Case Title: SAME title repeats for every step
Column 2 - Steps to Reproduce: Plain action sentences (NO step numbers)
Column 3 - Expected Result: "User should be able to..." / "User should not be able to..."
Column 4 - Actual Result: "User is able to..." / "User is not able to..."

STEP STRUCTURE FOR EVERY TEST CASE:
1. 7 mandatory login steps (always same)
2. 2-5 detailed navigation steps (generic but complete)
3. Final verification step from AC

Use ACTUAL Chiron element names:
- "Household appliances", "Marketing & Sales", "Orders"
- "Add to basket-B2B", "View Basket-B2B"
- "See Similar Products", "Explore Alternate"
- "Reference Product" (label with star icon)

🔥 FINAL REMINDER:
- Count AC points FIRST
- Generate AT MOST that many test cases
- ZERO extras, ZERO duplicates, ZERO "nice to have"
- ONLY direct AC verification
- Complete ALL test cases — never stop midway
- NEVER put "Step X:" prefix anywhere"""


def get_selenium_prompt(ac_text: str, tc_full_context: str = "", feature: str = "Feature") -> str:
    tc_section = f"""
═══════════════════════════════════════════════════════
🚨 USE THESE EXACT TEST CASES (DO NOT INVENT NEW ONES)
═══════════════════════════════════════════════════════

You MUST automate EVERY test case below — one @Test method per test case.
Convert each step into Selenium WebDriver actions.
Match the exact navigation flow shown in the steps.

{tc_full_context}

═══════════════════════════════════════════════════════
""" if tc_full_context else ""

    return f"""Act as a Senior Selenium Automation Engineer for AEG Chiron portal.

Generate COMPLETE Selenium Java automation code for:
Feature: {feature}

Acceptance Criteria:
{ac_text}
{tc_section}
{WEBSITE_KNOWLEDGE}

Generate ALL 4 files clearly separated:

===FILE: PageObject.java===
(Page Object Model — all @FindBy locators and action methods for ALL pages used in test cases)

===FILE: TestNGTest.java===
(TestNG test class — @BeforeClass with login, @Test method for EACH test case above, @AfterClass)

===FILE: testng.xml===
(TestNG suite XML config including all test methods)

===FILE: pom.xml===
(Maven pom with Selenium, TestNG, WebDriverManager dependencies)

🚨 CRITICAL RULES:
1. Generate ONE @Test method per test case from the test case list above
2. Method name should reflect the test case title (e.g., verifyUserAbleToSeeReferenceProduct())
3. Each @Test method must implement ALL the steps shown in that test case
4. Use TestNG Assert to validate the Expected Result for each step
5. Login steps go in @BeforeClass (don't repeat in each @Test)
6. Use real Chiron element names in locators (Add to basket-B2B, See Similar Products, Explore Alternate, Reference Product, etc.)
7. Login URL: https://t1-aeg-qa-a.eluxmkt.com/der/de/b2b/pre-login/
8. Use WebDriverManager for driver setup
9. Add meaningful comments referencing the test case being automated
10. Make code complete and production ready — NO placeholder TODOs

🚨 IMPORTANT: You MUST generate ALL 4 FILES (PageObject.java, TestNGTest.java, testng.xml, pom.xml).
Do NOT stop after 2 files. Keep code concise but complete to fit all 4 files."""


def get_bdd_prompt(ac_text: str, tc_full_context: str = "", feature: str = "Feature") -> str:
    fname = feature.replace(" ", "")

    tc_section = f"""
═══════════════════════════════════════════════════════
🚨 USE THESE EXACT TEST CASES (DO NOT INVENT NEW ONES)
═══════════════════════════════════════════════════════

You MUST convert EVERY test case below into a Cucumber Scenario.
Each test case = one Scenario in the .feature file.
Convert each step into Given/When/Then statements.

{tc_full_context}

═══════════════════════════════════════════════════════
""" if tc_full_context else ""

    return f"""Act as a Senior BDD Automation Engineer for AEG Chiron portal.

Generate COMPLETE Cucumber BDD automation code for:
Feature: {feature}

Acceptance Criteria:
{ac_text}
{tc_section}
{WEBSITE_KNOWLEDGE}

Generate ALL 4 files clearly separated:

===FILE: {fname}.feature===
(Gherkin feature file — Background with login steps, ONE Scenario per test case from list above)

===FILE: StepDefinitions.java===
(Java step definitions — EVERY Given/When/Then must have matching method)

===FILE: PageObject.java===
(Page Object Model — @FindBy locators for ALL pages used in scenarios)

===FILE: pom.xml===
(Maven pom with Selenium, Cucumber, TestNG, WebDriverManager dependencies)

🚨 CRITICAL RULES:
1. Generate ONE Scenario per test case from the test case list above
2. Scenario name should match the test case title
3. Each Scenario must implement ALL the steps from that test case
4. Background section should contain the 7 login steps (don't repeat in each scenario)
5. Use Given/When/Then keywords appropriately
6. Login URL: https://t1-aeg-qa-a.eluxmkt.com/der/de/b2b/pre-login/
7. Every step in feature file MUST have matching method in StepDefinitions.java
8. Use real Chiron element names (See Similar Products, Explore Alternate, Reference Product, etc.)
9. Add meaningful comments
10. Make code complete and production ready — NO placeholder TODOs

🚨 IMPORTANT: You MUST generate ALL 4 FILES (.feature, StepDefinitions.java, PageObject.java, pom.xml).
Do NOT stop after 2 files. Keep code concise but complete to fit all 4 files."""


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
                        # Clean up "Step X:" prefix if AI accidentally added it
                        import re
                        title = re.sub(r'^Step\s+\d+\s*:?\s*', '', title, flags=re.IGNORECASE).strip()
                        step = re.sub(r'^Step\s+\d+\s*:?\s*', '', step, flags=re.IGNORECASE).strip()
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
                        import re
                        title = re.sub(r'^Step\s+\d+\s*:?\s*', '', title, flags=re.IGNORECASE).strip()
                        step = re.sub(r'^Step\s+\d+\s*:?\s*', '', step, flags=re.IGNORECASE).strip()
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
                num_files = len(files)
                if num_files < 4:
                    st.warning(f"⚠️ Only {num_files} of 4 files generated. Try regenerating if needed.")
                else:
                    st.info(f"📦 Download ZIP contains all {num_files} files ready for your Java project!")
                st.download_button(
                    label=f"📦 Download All Files (ZIP) — {num_files} files",
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
                num_files = len(files)
                if num_files < 4:
                    st.warning(f"⚠️ Only {num_files} of 4 files generated. Try regenerating if needed.")
                else:
                    st.info(f"📦 Download ZIP contains all {num_files} files ready for your Java project!")
                st.download_button(
                    label=f"📦 Download All Files (ZIP) — {num_files} files",
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
st.sidebar.success("🌐 Chiron Website Knowledge: Loaded ✅")

# Show TC link status
if st.session_state.last_test_cases:
    unique_count = len(set(tc["Test Case Title"] for tc in st.session_state.last_test_cases))
    st.sidebar.info(f"🔗 {unique_count} TCs linked to Selenium/BDD")
else:
    st.sidebar.warning("⚠️ Generate TCs first for best Selenium/BDD output")

st.sidebar.markdown("---")

# 🤖 MODEL SELECTOR
st.sidebar.markdown("### 🤖 Choose AI Model")
selected_model = st.sidebar.selectbox(
    "Select Model",
    options=[
        "claude-haiku-4-5",
        "claude-sonnet-4-5",
        "claude-opus-4-5",
    ],
    format_func=lambda x: {
        "claude-haiku-4-5": "⚡ Haiku 4.5 (Fast)",
        "claude-sonnet-4-5": "⚖️ Sonnet 4.5 (Balanced)",
        "claude-opus-4-5": "🧠 Opus 4.5 (Smartest)",
    }[x],
    index=0,
    label_visibility="collapsed",
)
st.session_state.selected_model = selected_model
st.sidebar.caption(f"Currently using: **{selected_model}**")
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

    with st.spinner(f"🔍 Generating test cases using {st.session_state.selected_model}..."):
        reply = call_claude(
            messages=[{"role": "user", "content": prompt}],
            system=(
                "You are a QA expert who knows the AEG Chiron portal navigation deeply. "
                "🚨 STRICT 1:1 AC MAPPING — Count AC bullet points first. "
                "Generate AT MOST that many test cases. ZERO extras. "
                "🚨 Every Title MUST start with 'Verify whether user is able to' "
                "OR 'Verify whether user is not able to'. "
                "🚨 First show '📊 AC Analysis: Detected X AC points, Will generate X test cases'. "
                "Then show titles list, then full CSV between ---CSV START--- and ---CSV END---. "
                "🚨 NO 'logically needed' basics, NO 'nice to have' extras, NO duplicates. "
                "🚨 ONLY generate test cases that DIRECTLY map to an AC bullet point. "
                "🚨 Use generic navigation phrasing (any PLP category, any product with status...). "
                "🚨 NEVER put 'Step X:' prefix in step text or title text. "
                "🚨 Title column = ONLY title (NEVER step text). "
                "🚨 Step column = ONLY plain action sentence. "
                "Complete ALL test cases without stopping."
            ),
            max_tokens=TOKEN_BUDGETS["test_cases"],
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

    tc_full_context = build_full_tc_context(st.session_state.last_test_cases)
    has_tcs = bool(st.session_state.last_test_cases)

    prompt = get_selenium_prompt(ac_text, tc_full_context, feature)
    user_msg = f"**🤖 Generate Selenium TestNG Code for:** {feature}"
    if has_tcs:
        unique_count = len(set(tc["Test Case Title"] for tc in st.session_state.last_test_cases))
        user_msg += f"\n📋 Using {unique_count} test cases from CSV"
    push_block({"type": "chat", "role": "user", "content": user_msg})
    st.session_state.chat_history.append({"role": "user", "content": user_msg})

    spinner_msg = (
        f"⚙️ Generating Selenium TestNG (4 files, linked to {len(st.session_state.last_test_cases)} TC steps)..."
        if has_tcs else
        f"⚙️ Generating Selenium TestNG files using {st.session_state.selected_model}..."
    )

    with st.spinner(spinner_msg):
        reply = call_claude(
            messages=[{"role": "user", "content": prompt}],
            system=(
                "You are a Senior Selenium Automation Engineer who knows the AEG Chiron portal. "
                "Generate ALL 4 files separated by ===FILE: filename=== markers. "
                "Files: PageObject.java, TestNGTest.java, testng.xml, pom.xml. "
                "🚨 If test cases provided, generate ONE @Test method per test case. "
                "🚨 Each @Test method must implement ALL steps shown in that test case. "
                "🚨 Use TestNG Assert to validate Expected Result. "
                "🚨 Login steps go in @BeforeClass — do NOT repeat in each @Test. "
                "🚨 You MUST generate ALL 4 FILES — keep code concise but complete. "
                "Use real Chiron element names. Production ready."
            ),
            max_tokens=TOKEN_BUDGETS["selenium"],  # 21k = max safe limit
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

    with st.spinner(f"🔍 Step 1/2: Analyzing screenshot using {st.session_state.selected_model}..."):
        ui_description = call_claude(
            messages=[{"role": "user", "content": "Analyze this UI screenshot carefully and describe every UI element, button, field, text, and section you can see in detail."}],
            system="You are an expert UI analyst. Describe every UI element you see in the screenshot in complete detail.",
            images=st.session_state.images,
            max_tokens=4000,
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
                "You are a QA expert for AEG Chiron portal. "
                "🚨 STRICT 1:1 mapping — count AC/UI points, generate AT MOST that many TCs. "
                "🚨 ZERO extras, NO 'logically needed' basics. "
                "Every Title must start with 'Verify whether user is able to' or 'is not able to'. "
                "Show '📊 AC Analysis: Detected X points'. "
                "Then titles list, then CSV between markers. "
                "Generic navigation, detailed steps, NO 'Step X:' prefix anywhere."
            ),
            max_tokens=TOKEN_BUDGETS["screenshot"],
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

    tc_full_context = build_full_tc_context(st.session_state.last_test_cases)
    has_tcs = bool(st.session_state.last_test_cases)

    prompt = get_bdd_prompt(ac_text, tc_full_context, feature)
    user_msg = f"**📝 Generate BDD Cucumber Code for:** {feature}"
    if has_tcs:
        unique_count = len(set(tc["Test Case Title"] for tc in st.session_state.last_test_cases))
        user_msg += f"\n📋 Using {unique_count} test cases from CSV"
    push_block({"type": "chat", "role": "user", "content": user_msg})
    st.session_state.chat_history.append({"role": "user", "content": user_msg})

    spinner_msg = (
        f"📝 Generating BDD Cucumber (4 files, linked to {len(st.session_state.last_test_cases)} TC steps)..."
        if has_tcs else
        f"📝 Generating BDD Cucumber files using {st.session_state.selected_model}..."
    )

    with st.spinner(spinner_msg):
        reply = call_claude(
            messages=[{"role": "user", "content": prompt}],
            system=(
                "You are a Senior BDD Automation Engineer who knows AEG Chiron portal. "
                "Generate ALL 4 files separated by ===FILE: filename=== markers. "
                "Files: .feature file, StepDefinitions.java, PageObject.java, pom.xml. "
                "🚨 If test cases provided, generate ONE Scenario per test case. "
                "🚨 Each Scenario must implement ALL steps from that test case. "
                "🚨 Background must contain 7 login steps — don't repeat in scenarios. "
                "🚨 You MUST generate ALL 4 FILES — keep code concise but complete. "
                "Every step in feature file MUST have matching method in StepDefinitions.java. "
                "Production ready."
            ),
            max_tokens=TOKEN_BUDGETS["bdd"],  # 21k = max safe limit
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

    with st.spinner(f"📄 Generating report using {st.session_state.selected_model}..."):
        reply = call_claude(
            messages=[{"role": "user", "content": prompt}],
            system="You are a QA Test Lead writing professional reports for QA Managers. Be concise and professional.",
            max_tokens=TOKEN_BUDGETS["summary"],
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
        "You are an expert QA Engineer and Technical Test Lead who knows the AEG Chiron portal navigation. "
        "Help with test cases, Selenium Java, bug reports, and QA best practices. "
        "Expected Result = User should be able to... "
        "Actual Result = User is able to..."
    )
    if st.session_state.file_text:
        system_msg += f"\n\nDoc context:\n{st.session_state.file_text[:4000]}"
    if st.session_state.last_ac:
        system_msg += f"\n\nPrevious AC:\n{st.session_state.last_ac}"

    raw_history = st.session_state.chat_history[-10:]
    api_messages = sanitize_messages([
        {"role": m["role"], "content": m["content"]}
        for m in raw_history
    ])

    if not api_messages or api_messages[-1]["role"] != "user":
        api_messages.append({"role": "user", "content": user_prompt})

    use_image = st.session_state.images and any(
        w in user_prompt.lower()
        for w in ["image", "screenshot", "screen", "this", "describe", "analyze"]
    )

    with st.spinner(f"Thinking using {st.session_state.selected_model}..."):
        reply = call_claude(
            messages=api_messages,
            system=system_msg,
            images=(st.session_state.images if use_image else None),
            max_tokens=TOKEN_BUDGETS["free_chat"],
        )

    push_block({"type": "chat", "role": "assistant", "content": reply})
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.rerun()
