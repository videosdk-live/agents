"""
demo_multilang.py
─────────────────────────────────────────────────────────────────────────────
One loan-advisor agent, four languages — English / Hindi / Gujarati / Marathi.

Pick the language as the first CLI argument (default "hi"):

    python demo_multilang.py hi
    python demo_multilang.py gu
    python demo_multilang.py mr
    python demo_multilang.py en

Providers:
  • STT  — Deepgram Nova-3   (required for hi/gu/mr; supports en too)
  • LLM  — Google Gemini 2.5 Flash
  • TTS  — Cartesia Sonic-3  (required for hi/gu/mr; supports en too)

"""

from __future__ import annotations

import logging
import sys

from videosdk.agents import (
    Agent,
    AgentSession,
    EOUConfig,
    InterruptConfig,
    JobContext,
    Pipeline,
    RoomOptions,
    WorkerJob,
    function_tool,
    pre_warm_tokenizer, 
)
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model


_arg = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "hi"
LANG = _arg if _arg in {"en", "hi", "gu", "mr"} else "hi"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
# Uncomment to trace LLM delta → filter → chunk → TTS:
# logging.getLogger("videosdk.agents.tokenize.filters").setLevel(logging.DEBUG)
# logging.getLogger("videosdk.agents.tokenize.stream").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

pre_download_model()
if LANG != "en":
    pre_warm_tokenizer()

_EN_PROMPT = """
You are a female English-speaking loan advisor for small business owners,
so pronounce the words clearly, professionally, and in a friendly manner.
Keep sentences short and conversational — two or three ideas per turn.
Speak numbers naturally: "$1,200 a month", "twelve percent a year".
Use titles: Dr. Smith, Mr. Johnson, Prof. Lee.
When a tool returns data, weave the numbers into spoken sentences — never
read raw JSON. Acknowledge the user, give the information, ask one follow-up.
"""

_HI_PROMPT = """
आप छोटे व्यवसाय मालिकों के लिए एक महिला इंग्लिश बोलने वाली लोन एडवाइजर हैं,
इसलिए शब्दों का उच्चारण स्पष्ट, प्रोफेशनल और मित्रतापूर्ण तरीके से करें।
छोटे, natural वाक्य बोलें — एक बार में दो-तीन से ज़्यादा बातें नहीं।
संख्याएँ स्वाभाविक रूप से बोलें: "₹50,000 प्रति माह", "बारह प्रतिशत प्रति वर्ष"।
Honorifics use करें: डॉ. शर्मा, श्री. गुप्ता, प्रो. रमेश।
Technical terms (EMI, CIBIL, GST) English में बोलें।
जब tool का result आए तो उसे natural Hindi में explain करें — raw JSON मत पढ़ें।
"""

_GU_PROMPT = """
तुम्ही लहान व्यवसाय मालकांसाठी महिला इंग्रजी बोलणारी लोन अॅडव्हायझर आहात,
त्यामुळे शब्दांचा उच्चार स्पष्ट, व्यावसायिक आणि मैत्रीपूर्ण पद्धतीने करा.”
તેથી શબ્દોનું ઉચ્ચારણ સ્પષ્ટ, પ્રોફેશનલ અને મિત્રતાપૂર્વક કરો.”
ટૂંકા, natural વાક્યો બોલો — એક સાથે બે-ત્રણ થી વધુ વાત નહીં.
સંખ્યાઓ સ્વાભાવિક રીતે બોલો: "₹50,000 પ્રતિ મહિને", "બાર ટકા પ્રતિ વર્ષ".
Honorifics: ડૉ. શર્મા, શ્રી. ગુપ્તા, પ્રો. રમેશ.
Technical terms (EMI, CIBIL, GST) English માં બોલો.
Tool નું result natural Gujarati માં સમજાવો — raw JSON ન વાંચો.
"""

_MR_PROMPT = """
तुम्ही एक मुलगी Marathi loan advisor आहात जे small business owners ना मदत करतात.
छोटी, natural वाक्ये बोला — एका वेळी दोन-तीनपेक्षा जास्त मुद्दे नकोत.
संख्या नैसर्गिकपणे बोला: "₹50,000 दरमहा", "बारा टक्के दरवर्षी".
Honorifics: डॉ. शर्मा, श्री. गुप्ता, प्रो. रमेश.
Technical terms (EMI, CIBIL, GST) English मध्ये बोला.
Tool चा result natural Marathi मध्ये समजावा — raw JSON वाचू नका.
"""

LANGUAGES: dict[str, dict] = {
    "en": {
        "label": "English",
        "deepgram_language": "en",
        "cartesia_language": "en",
        "currency": "$ (US dollars)",
        "greeting": "Hi! I'm your business loan advisor. Want to hear about our loan products, check eligibility, or work out an EMI?",
        "instructions": _EN_PROMPT,
    },
    "hi": {
        "label": "Hindi",
        "deepgram_language": "hi",
        "cartesia_language": "hi",
        "currency": "₹ (rupees)",
        "greeting": "नमस्ते! मैं आपकी business loan advisor हूँ। आप loan products, eligibility, या EMI के बारे में पूछ सकते हैं — किससे शुरू करें?",
        "instructions": _HI_PROMPT,
    },
    "gu": {
        "label": "Gujarati",
        "deepgram_language": "gu",
        "cartesia_language": "gu",
        "currency": "₹ (rupees)",
        "greeting": "નમસ્તે! હું તમારી business loan advisor છું. તમે loan products, eligibility, કે EMI વિશે પૂછી શકો છો — શેનાથી શરૂ કરીએ?",
        "instructions": _GU_PROMPT,
    },
    "mr": {
        "label": "Marathi",
        "deepgram_language": "mr",
        "cartesia_language": "mr",
        "currency": "₹ (rupees)",
        "greeting": "नमस्कार! मी तुमची business loan advisor आहे. तुम्ही loan products, eligibility, किंवा EMI बद्दल विचारू शकता — कशापासून सुरुवात करूया?",
        "instructions": _MR_PROMPT,
    },
}

CFG = LANGUAGES[LANG]


LOAN_PRODUCTS = {
    "business": {
        "name": "Business Growth Loan",
        "amount_range": "100,000 - 5,000,000",
        "rate": "12% - 18% per year",
        "tenure_months": "12 - 60",
        "processing_fee": "2% + tax",
    },
    "working_capital": {
        "name": "Working Capital Loan",
        "amount_range": "50,000 - 1,000,000",
        "rate": "14% - 20% per year",
        "tenure_months": "6 - 24",
        "processing_fee": "1.5% + tax",
    },
    "msme": {
        "name": "MSME / Small Business Loan",
        "amount_range": "10,000 - 2,500,000",
        "rate": "7% - 10% per year (subsidised)",
        "tenure_months": "12 - 84",
        "processing_fee": "0",
    },
}


@function_tool
async def get_loan_products(loan_type: str) -> dict:
    """Return details for a loan product. `loan_type` is one of:
    "business", "working_capital", "msme". Amounts are in the agent's currency."""
    product = LOAN_PRODUCTS.get(loan_type.lower())
    if not product:
        return {"error": f"unknown loan_type {loan_type!r}", "available": list(LOAN_PRODUCTS)}
    return {**product, "currency": CFG["currency"]}


@function_tool
async def calculate_emi(principal: float, annual_rate_percent: float, tenure_months: int) -> dict:
    """Compute the monthly EMI for a loan. `principal` is in the agent's currency."""
    r = annual_rate_percent / 12 / 100
    if r == 0:
        emi = principal / tenure_months
    else:
        emi = principal * r * (1 + r) ** tenure_months / ((1 + r) ** tenure_months - 1)
    total = emi * tenure_months
    return {
        "currency": CFG["currency"],
        "monthly_emi": round(emi),
        "total_payment": round(total),
        "total_interest": round(total - principal),
        "tenure_months": tenure_months,
        "annual_rate_percent": annual_rate_percent,
    }


@function_tool
async def check_eligibility(cibil_score: int, business_age_years: float, monthly_turnover: float) -> dict:
    """Check basic loan eligibility. `monthly_turnover` is in the agent's currency."""
    issues = []
    if cibil_score < 650:
        issues.append(f"CIBIL score {cibil_score} is below the minimum 650")
    if business_age_years < 2:
        issues.append(f"business is {business_age_years} years old, need at least 2")
    if monthly_turnover < 50_000:
        issues.append(f"monthly turnover {monthly_turnover} is below the minimum 50,000")
    return {"eligible": not issues, "issues": issues}


class MultilangLoanAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=CFG["instructions"],
            tools=[get_loan_products, calculate_emi, check_eligibility],
        )

    async def on_enter(self) -> None:
        await self.session.say(CFG["greeting"])

    async def on_exit(self) -> None:
        bye = {
            "en": "Thanks for stopping by — talk soon!",
            "hi": "धन्यवाद! फिर मिलते हैं।",
            "gu": "આભાર! ફરી મળીશું.",
            "mr": "धन्यवाद! पुन्हा भेटूया.",
        }[LANG]
        await self.session.say(bye)


async def entrypoint(ctx: JobContext):
    logger.info("Starting multilingual loan agent | LANG=%s", LANG)

    agent = MultilangLoanAgent()
    pipeline = Pipeline(
        stt=DeepgramSTT(model="nova-3", language=CFG["deepgram_language"]),
        llm=GoogleLLM(model="gemini-2.5-flash"),
        tts=CartesiaTTS(model="sonic-3", language=CFG["cartesia_language"]),
        vad=SileroVAD(threshold=0.8),
        turn_detector=TurnDetector(),
        eou_config=EOUConfig(mode="DEFAULT", min_max_speech_wait_timeout=[0.1, 0.2]),
        interrupt_config=InterruptConfig(
            interrupt_min_duration=0.2,
            interrupt_min_words=2,
            interrupt_min_confidence=0.6,
            false_interrupt_pause_duration=2.0,
            resume_on_false_interrupt=True,
        ),
    )

    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    return JobContext(
        room_options=RoomOptions(
            room_id="<room_id>",
            name=f"Multilingual Loan Agent ({CFG['label']})",
            playground=True,
        )
    )

if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()