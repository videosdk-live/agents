"""
travel_agency.py
================
A comprehensive ~50-node travel agency workflow powered by LLM + STT + TTS.

Covers the FULL travel booking lifecycle:
  - Customer onboarding (name, phone, email)
  - Trip type detection (domestic / international)
  - Trip planning (origin, destination, dates, travelers, budget)
  - Transport booking with 3 branches (train / flight / bus)
  - Hotel booking (location, room type, extras)
  - Day-by-day itinerary generation
  - Visa & passport assistance (international only)
  - Travel insurance
  - Payment via Razorpay (with HumanInLoop webhook)
  - Booking confirmation, SMS, feedback

Flow (simplified)
-----------------
START → welcome → collect_name → collect_phone → collect_email
      → detect_trip_type
      → collect_origin → collect_destination → collect_departure_date
      → collect_return_date → collect_num_travelers → collect_budget
      → select_transport
          ├─ TRAIN: search_trains → select_train → train_class → train_berth → confirm_train ─┐
          ├─ FLIGHT: search_flights → select_flight → flight_class → flight_seat              │
          │          → flight_meal → confirm_flight------------------------------------------─┤
          └─ BUS: search_buses → select_bus → bus_seat → confirm_bus ─────────────────────────┘
                                                                                               │
      ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← (merge) ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←┘
      → ask_hotel → collect_hotel_location → collect_checkin_checkout → collect_room_type
      → collect_num_rooms → search_hotels → select_hotel → hotel_extras → confirm_hotel
      → collect_interests → generate_itinerary → review_itinerary
      → [international?] → check_visa → collect_passport → visa_info
      → offer_insurance → select_insurance
      → order_summary → select_payment → initiate_payment
          ├─ payment_success → send_confirmation → collect_feedback → farewell → END
          └─ payment_failed → END

Total: 51 nodes
"""

import asyncio
import logging
import os
import random
import string
import time

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

from conversational_graph import (
    ConversationalGraph,
    GraphState,
    Context,
    Route,
    END,
    START,
    Interrupt,
    HumanInLoop,
    GraphConfig,
    FileSaver,
    LatencyProfilerHook,
)

from videosdk.agents import Agent, Pipeline, AgentSession, JobContext, WorkerJob, RoomOptions
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.cartesia import CartesiaTTS

from pydantic import Field, field_validator
from typing import Optional


#   1. STATE MODEL                                                              
class TravelState(GraphState):
    #  Customer
    customer_name:    Optional[str]  = Field(None, description="Customer full name")
    phone:            Optional[str]  = Field(None, description="Contact phone (10+ digits)")
    email:            Optional[str]  = Field(None, description="Customer email address")

    #  Trip basics 
    trip_type:        Optional[str]  = Field(None, description="domestic or international")
    origin:           Optional[str]  = Field(None, description="Origin city")
    destination:      Optional[str]  = Field(None, description="Destination city or country")
    departure_date:   Optional[str]  = Field(None, description="Departure date YYYY-MM-DD")
    return_date:      Optional[str]  = Field(None, description="Return date YYYY-MM-DD")
    num_travelers:    Optional[int]  = Field(None, ge=1, le=30, description="Number of travelers")
    budget_tier:      Optional[str]  = Field(None, description="budget, moderate, or luxury")

    #  Transport 
    transport_mode:   Optional[str]  = Field(None, description="train, flight, or bus")
    train_name:       Optional[str]  = Field(None, description="Selected train name/number")
    train_class:      Optional[str]  = Field(None, description="sleeper, 3ac, 2ac, 1ac, or chair_car")
    berth_preference: Optional[str]  = Field(None, description="lower, middle, upper, side_lower, side_upper, or no_preference")
    flight_name:      Optional[str]  = Field(None, description="Selected flight airline + number")
    flight_class:     Optional[str]  = Field(None, description="economy, premium_economy, business, or first")
    seat_preference:  Optional[str]  = Field(None, description="window, middle, or aisle")
    meal_preference:  Optional[str]  = Field(None, description="veg, non_veg, vegan, jain, or no_meal")
    bus_name:         Optional[str]  = Field(None, description="Selected bus operator + type")
    bus_seat_type:    Optional[str]  = Field(None, description="seater, sleeper, or semi_sleeper")
    transport_confirmed: Optional[bool] = Field(None, description="Transport booking confirmed")

    #  Hotel 
    needs_hotel:      Optional[bool] = Field(None, description="Whether user wants hotel booking")
    hotel_location:   Optional[str]  = Field(None, description="Preferred hotel area/locality")
    checkin_date:     Optional[str]  = Field(None, description="Hotel check-in date YYYY-MM-DD")
    checkout_date:    Optional[str]  = Field(None, description="Hotel check-out date YYYY-MM-DD")
    room_type:        Optional[str]  = Field(None, description="single, double, twin, suite, or dormitory")
    num_rooms:        Optional[int]  = Field(None, ge=1, le=20, description="Number of rooms")
    hotel_name:       Optional[str]  = Field(None, description="Selected hotel name")
    hotel_extras:     Optional[str]  = Field(None, description="Extra requests: breakfast, airport_pickup, spa, etc.")
    hotel_confirmed:  Optional[bool] = Field(None, description="Hotel booking confirmed")

    #  Itinerary 
    interests:        Optional[str]  = Field(None, description="Comma-separated travel interests")
    itinerary_text:   Optional[str]  = Field(None, description="Generated day-by-day itinerary")
    itinerary_approved: Optional[bool] = Field(None, description="User approved the itinerary")

    #  Visa & Passport (international) 
    visa_required:    Optional[bool] = Field(None, description="Whether visa is required for destination")
    passport_number:  Optional[str]  = Field(None, description="Passport number")
    passport_expiry:  Optional[str]  = Field(None, description="Passport expiry date YYYY-MM-DD")
    visa_status:      Optional[str]  = Field(None, description="not_needed, have_visa, need_visa")

    #  Insurance 
    wants_insurance:  Optional[bool] = Field(None, description="User wants travel insurance")
    insurance_plan:   Optional[str]  = Field(None, description="basic, standard, or premium")

    #  Payment 
    total_amount:     Optional[int]  = Field(None, description="Total booking amount in INR")
    payment_method:   Optional[str]  = Field(None, description="upi, card, netbanking, or wallet")
    payment_status:   Optional[str]  = Field(None, description="pending, paid, or failed")
    payment_id:       Optional[str]  = Field(None, description="Payment transaction ID")
    payment_link:     Optional[str]  = Field(None, description="Payment URL")

    #  Confirmation 
    booking_ref:      Optional[str]  = Field(None, description="Master booking reference code")
    feedback_rating:  Optional[int]  = Field(None, ge=1, le=5, description="User rating 1-5")
    feedback_comment: Optional[str]  = Field(None, description="Optional feedback text")

    #  Validators 
    @field_validator("trip_type")
    @classmethod
    def _v_trip_type(cls, v):
        if v and v.lower().strip() not in {"domestic", "international"}:
            raise ValueError("Must be domestic or international")
        return v.lower().strip() if v else v

    @field_validator("budget_tier")
    @classmethod
    def _v_budget(cls, v):
        if not v:
            return v
        aliases = {
            "cheap": "budget", "low": "budget", "budget": "budget", "backpacker": "budget",
            "mid": "moderate", "moderate": "moderate", "medium": "moderate", "standard": "moderate",
            "luxury": "luxury", "high": "luxury", "premium": "luxury", "expensive": "luxury",
        }
        n = aliases.get(v.lower().strip())
        if not n:
            raise ValueError("Must be budget, moderate, or luxury")
        return n

    @field_validator("transport_mode")
    @classmethod
    def _v_transport(cls, v):
        if v and v.lower().strip() not in {"train", "flight", "bus"}:
            raise ValueError("Must be train, flight, or bus")
        return v.lower().strip() if v else v

    @field_validator("phone")
    @classmethod
    def _v_phone(cls, v):
        if v:
            digits = "".join(filter(str.isdigit, v))
            if len(digits) < 10:
                raise ValueError("Phone must be at least 10 digits")
            return digits
        return v

    @field_validator("payment_method")
    @classmethod
    def _v_payment(cls, v):
        if v and v.lower().strip() not in {"upi", "card", "netbanking", "wallet"}:
            raise ValueError("Must be upi, card, netbanking, or wallet")
        return v.lower().strip() if v else v


#   2. PER-NODE EXTRACTION SCHEMAS                                              


class NameSchema(GraphState):
    customer_name: str = Field(..., description="Customer full name")

class PhoneSchema(GraphState):
    phone: str = Field(..., description="Phone number (10+ digits)")

class EmailSchema(GraphState):
    email: str = Field(..., description="Email address")

class TripTypeSchema(GraphState):
    trip_type: str = Field(..., description="domestic or international")

class OriginSchema(GraphState):
    origin: str = Field(..., description="Origin city name")

class DestinationSchema(GraphState):
    destination: str = Field(..., description="Destination city or country")

class TransportModeSchema(GraphState):
    transport_mode: str = Field(..., description="train, flight, or bus")

class TrainClassSchema(GraphState):
    train_class: str = Field(..., description="sleeper, 3ac, 2ac, 1ac, or chair_car")

class BerthSchema(GraphState):
    berth_preference: Optional[str] = Field(None, description="lower, middle, upper, side_lower, side_upper, or no_preference")

class FlightClassSchema(GraphState):
    flight_class: str = Field(..., description="economy, premium_economy, business, or first")

class SeatSchema(GraphState):
    seat_preference: Optional[str] = Field(None, description="window, middle, or aisle")

class MealSchema(GraphState):
    meal_preference: Optional[str] = Field(None, description="veg, non_veg, vegan, jain, or no_meal")

class BusSeatSchema(GraphState):
    bus_seat_type: Optional[str] = Field(None, description="seater, sleeper, or semi_sleeper")

class RoomTypeSchema(GraphState):
    room_type: str = Field(..., description="single, double, twin, suite, or dormitory")

class InsurancePlanSchema(GraphState):
    insurance_plan: str = Field(..., description="basic, standard, or premium")

class PaymentMethodSchema(GraphState):
    payment_method: str = Field(..., description="upi, card, netbanking, or wallet")

class FeedbackSchema(GraphState):
    feedback_rating:  int = Field(..., ge=1, le=5, description="Rating 1-5")
    feedback_comment: Optional[str] = Field(None, description="Optional feedback")


#   3. HELPERS                                                                   

def _gen_ref() -> str:
    return "TRV-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


def _estimate_cost(state: TravelState) -> int:
    """Rough cost estimator based on selections."""
    base = {"budget": 5000, "moderate": 15000, "luxury": 40000}.get(state.budget_tier or "moderate", 15000)
    transport = {"train": 800, "flight": 4500, "bus": 500}.get(state.transport_mode or "train", 800)
    hotel = {"single": 2000, "double": 3000, "twin": 3000, "suite": 8000, "dormitory": 800}.get(state.room_type or "double", 3000)
    insurance = {"basic": 500, "standard": 1200, "premium": 2500, "none": 0}.get(state.insurance_plan or "none", 0)
    travelers = state.num_travelers or 1
    rooms = state.num_rooms or 1
    return (base + (transport * travelers) + (hotel * rooms) + (insurance * travelers))


INDIAN_CITIES = [
    "Mumbai",
    "Delhi",
    "Bangalore",
    "Hyderabad",
    "Chennai",
    "Kolkata",
    "Pune",
    "Ahmedabad",
    "Jaipur",
    "Surat",
    "Bhopal",
    "Goa"
]
# ── Static catalog data (replace with real API calls in production) ───

TRAINS = [
    {"name": "Rajdhani Express", "number": "12951", "depart": "4:25 PM", "arrive": "8:35 AM", "duration": "16h 10m", "fare_sl": 650, "fare_3ac": 1700, "fare_2ac": 2500, "fare_1ac": 4200},
    {"name": "Shatabdi Express", "number": "12002", "depart": "6:00 AM", "arrive": "2:15 PM", "duration": "8h 15m", "fare_sl": 550, "fare_3ac": 1400, "fare_2ac": 2100, "fare_1ac": 3600},
    {"name": "Duronto Express",  "number": "12267", "depart": "8:30 PM", "arrive": "9:45 AM", "duration": "13h 15m", "fare_sl": 600, "fare_3ac": 1600, "fare_2ac": 2300, "fare_1ac": 3900},
    {"name": "Garib Rath",       "number": "12216", "depart": "5:15 PM", "arrive": "7:00 AM", "duration": "13h 45m", "fare_sl": 450, "fare_3ac": 1100, "fare_2ac": None, "fare_1ac": None},
]

FLIGHTS = [
    {"airline": "IndiGo",    "number": "6E-2451", "depart": "6:30 AM", "arrive": "8:45 AM", "duration": "2h 15m", "fare_eco": 4500, "fare_biz": 12000},
    {"airline": "Air India",  "number": "AI-806",  "depart": "9:15 AM", "arrive": "11:30 AM", "duration": "2h 15m", "fare_eco": 5200, "fare_biz": 14500},
    {"airline": "Vistara",    "number": "UK-963",  "depart": "2:00 PM", "arrive": "4:10 PM", "duration": "2h 10m", "fare_eco": 5800, "fare_biz": 15000},
    {"airline": "SpiceJet",   "number": "SG-8129", "depart": "7:45 PM", "arrive": "10:00 PM", "duration": "2h 15m", "fare_eco": 3900, "fare_biz": 10500},
]

BUSES = [
    {"operator": "VRL Travels",      "type": "Volvo AC Sleeper",    "depart": "8:00 PM", "arrive": "8:00 AM", "duration": "12h", "fare": 1200},
    {"operator": "SRS Travels",      "type": "Non-AC Seater",       "depart": "9:30 PM", "arrive": "10:00 AM", "duration": "12h 30m", "fare": 600},
    {"operator": "Orange Travels",   "type": "Volvo AC Semi-Sleeper", "depart": "10:00 PM", "arrive": "9:30 AM", "duration": "11h 30m", "fare": 950},
    {"operator": "Neeta Travels",    "type": "AC Sleeper",          "depart": "11:00 PM", "arrive": "10:30 AM", "duration": "11h 30m", "fare": 1100},
]

HOTELS = {
    "budget": [
        {"name": "Hotel Sunrise",      "stars": 2, "price": 1200, "amenities": "WiFi, AC, TV",                    "distance": "1.5 km from center"},
        {"name": "Backpacker's Den",    "stars": 1, "price": 600,  "amenities": "WiFi, shared bathroom",           "distance": "2 km from center"},
        {"name": "OYO Rooms Express",   "stars": 2, "price": 900,  "amenities": "WiFi, AC, breakfast",             "distance": "3 km from center"},
    ],
    "moderate": [
        {"name": "Hotel Grand Plaza",   "stars": 3, "price": 3500, "amenities": "WiFi, AC, pool, restaurant",     "distance": "500m from center"},
        {"name": "Comfort Inn Suites",  "stars": 3, "price": 3000, "amenities": "WiFi, AC, gym, breakfast",        "distance": "1 km from center"},
        {"name": "Treebo Trend Royal",  "stars": 3, "price": 2800, "amenities": "WiFi, AC, restaurant, parking",   "distance": "1.2 km from center"},
    ],
    "luxury": [
        {"name": "Taj Gateway",         "stars": 5, "price": 9500, "amenities": "WiFi, pool, spa, fine dining, gym, concierge", "distance": "city center"},
        {"name": "ITC Grand",           "stars": 5, "price": 11000, "amenities": "WiFi, pool, spa, 3 restaurants, butler service", "distance": "city center"},
        {"name": "The Leela Palace",    "stars": 5, "price": 13500, "amenities": "WiFi, pool, spa, fine dining, rooftop bar",    "distance": "city center"},
    ],
}


def _format_trains(origin: str, dest: str, date: str, travelers: int) -> str:
    lines = [f"Here are the trains I found from {origin} to {dest} on {date}:\n"]
    for i, t in enumerate(TRAINS, 1):
        lines.append(
            f"  {i}. {t['name']} ({t['number']}) — leaves at {t['depart']}, "
            f"arrives {t['arrive']} ({t['duration']}). "
            f"Fare per person: SL ₹{t['fare_sl']}, 3AC ₹{t['fare_3ac']}"
            + (f", 2AC ₹{t['fare_2ac']}" if t['fare_2ac'] else "")
            + (f", 1AC ₹{t['fare_1ac']}" if t['fare_1ac'] else "")
        )
    lines.append(f"\nWhich one works for you?")
    return "\n".join(lines)


def _format_flights(origin: str, dest: str, date: str, travelers: int) -> str:
    lines = [f"Here are the flights from {origin} to {dest} on {date}:\n"]
    for i, f in enumerate(FLIGHTS, 1):
        lines.append(
            f"  {i}. {f['airline']} {f['number']} — {f['depart']} to {f['arrive']} "
            f"({f['duration']}). Economy ₹{f['fare_eco']}, Business ₹{f['fare_biz']}"
        )
    lines.append(f"\nWhich flight do you like?")
    return "\n".join(lines)


def _format_buses(origin: str, dest: str, date: str) -> str:
    lines = [f"Here are the buses from {origin} to {dest} on {date}:\n"]
    for i, b in enumerate(BUSES, 1):
        lines.append(
            f"  {i}. {b['operator']} — {b['type']}, leaves {b['depart']}, "
            f"arrives {b['arrive']} ({b['duration']}). Fare: ₹{b['fare']}"
        )
    lines.append(f"\nWhich one sounds good?")
    return "\n".join(lines)


def _format_hotels(dest: str, budget: str, location: str, rooms: int, room_type: str) -> str:
    tier = HOTELS.get(budget or "moderate", HOTELS["moderate"])
    lines = [f"Here are some {budget} hotels near {location} in {dest}:\n"]
    for i, h in enumerate(tier, 1):
        lines.append(
            f"  {i}. {h['name']} ({'⭐' * h['stars']}) — ₹{h['price']}/night. "
            f"{h['amenities']}. {h['distance']}"
        )
    lines.append(f"\nWhich one do you like?")
    return "\n".join(lines)


# ── Razorpay / SMS config

RAZORPAY_KEY_ID        = os.environ.get("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET    = os.environ.get("RAZORPAY_KEY_SECRET", "")
RAZORPAY_WEBHOOK_SECRET = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")
WEBHOOK_PORT           = int(os.environ.get("WEBHOOK_PORT", "8080"))
PAYMENT_LINK_EXPIRE_MIN = int(os.environ.get("PAYMENT_LINK_EXPIRE_MIN", "30"))

TWILIO_ACCOUNT_SID  = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN   = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER  = os.environ.get("TWILIO_FROM_NUMBER", "")


async def _create_payment_link(ref: str, amount: int, name: str, phone: str) -> str:
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        return f"http://localhost:{WEBHOOK_PORT}/mock-pay?ref={ref}&amount={amount}"
    import httpx
    to_e164 = phone if phone.startswith("+") else f"+91{phone}"
    payload = {
        "amount": amount * 100, "currency": "INR", "accept_partial": False,
        "description": f"Travel booking {ref}",
        "customer": {"name": name, "contact": to_e164},
        "notify": {"sms": True, "email": False},
        "reference_id": ref,
        "expire_by": int(time.time()) + PAYMENT_LINK_EXPIRE_MIN * 60,
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://api.razorpay.com/v1/payment_links",
            json=payload, auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET),
        )
        resp.raise_for_status()
    return resp.json()["short_url"]


async def _send_sms(to: str, body: str) -> None:
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, to]):
        logger.warning("[SMS] Twilio not configured — skipping")
        return
    from twilio.rest import Client
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    digits = "".join(filter(str.isdigit, to))
    to_e164 = to if to.startswith("+") else f"+91{digits}"
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, lambda: client.messages.create(body=body, from_=TWILIO_FROM_NUMBER, to=to_e164),
        )



#   4. GRAPH CONFIG                                                              
config = GraphConfig(
    name="travel_agency",
    graph_id="TRAVEL_AGENCY_001",
    debug=True,
    checkpointer=FileSaver(directory="./checkpoints"),
    max_retries=10,
    stream=True
)

graph = ConversationalGraph(TravelState, config)

#   5. NODES  (51 total)                                                                                                                                    
#   Naming: section_action  (e.g. transport_search_trains)                       


# ── 5A. CUSTOMER ONBOARDING (nodes 1-4) ──

async def welcome(state: TravelState, ctx: Context):                              # 1
    """Entry point — greet and start collecting info."""
    return Route("collect_name")


async def collect_name(state: TravelState, ctx: Context):                          # 2
    if state.customer_name:
        await ctx.say(f"Nice to meet you, {state.customer_name}!")
        return Route("collect_phone")

    result = await ctx.extractor.collect(
        schema=NameSchema,
        prompt=(
            "Hey, welcome to Yatra Express! I'm going to help you plan your whole trip today. "
            "If the user already said their name, greet them by name and move on. "
            "Otherwise, ask for their name."
        ),
    )
    if result.extracted.customer_name:
        await ctx.say(f"Nice to meet you, {result.extracted.customer_name}. Can you please share your phone number?")
        return Route("collect_phone")

    return Interrupt(
        say=(
            "If the user already said their name in the conversation, extract it and greet them. "
            "Otherwise ask what their name is. Do NOT apologize."
        ),
        id="retry_name",
    )


async def collect_phone(state: TravelState, ctx: Context):                         # 3
    if state.phone:
        return Route("collect_email")

    result = await ctx.extractor.collect(
        schema=PhoneSchema,
        prompt=(
            f"I'll need your phone number to proceed, {state.customer_name}. "
            "We can't move forward with the booking without it — it's required "
            "for sending the payment link and booking confirmation."
        ),
    )
    if result.extracted.phone:
        return Route("collect_email")
    return Interrupt(
        say=(
            f"Can you please share your phone number,{state.customer_name}? The phone number "
            "is mandatory — we cannot proceed without it. It's only used for sending "
            "your booking confirmation and payment link. Could you please share it?"
        ),
        id="retry_phone",
    )


async def collect_email(state: TravelState, ctx: Context):                         # 4
    if state.email:
        return Route("detect_trip_type")

    result = await ctx.extractor.collect(
        schema=EmailSchema,
        prompt="And your email? That's where I'll send the e-tickets and the full itinerary.",
    )
    if result.extracted.email:
        return Route("detect_trip_type")

    return Interrupt(
        say=(
            "can you share your email address to send e-tickets and the itinerary. "
        ),
        id="retry_email",
    )


# ── 5B. TRIP TYPE & PLANNING (nodes 5-11) 

async def detect_trip_type(state: TravelState, ctx: Context):                      # 5
    """Detect domestic vs international."""
    trip = state.trip_type

    # Auto-detect from destination if already known
    if not trip and state.destination:
        if state.destination.lower().strip() in INDIAN_CITIES:
            trip = "domestic"
        else:
            trip = "international"

    if not trip:
        intent = await ctx.extractor.match_intent(
            intents={
                "domestic":      "traveling within India, domestic trip, local, same country, "
                                 + ", ".join(list(INDIAN_CITIES)[:20]),
                "international": "traveling abroad, international, foreign country, overseas, visa, "
                                 "dubai, singapore, thailand, europe, usa, london, paris, bali",
            }
        )
        if intent in ("domestic", "international"):
            trip = intent

    if trip == "domestic":
        return Route("collect_origin", update={"trip_type": "domestic"})
    if trip == "international":
        return Route("collect_origin", update={"trip_type": "international"})
    return Interrupt(
        say=(
            f"Ask {state.customer_name} if they're planning a trip within India (domestic) "
            "or going abroad (international). If they already mentioned a city or country, "
            "figure out which one it is."
        ),
        id="retry_trip_type",
    )


async def collect_origin(state: TravelState, ctx: Context):                        # 6
    result = await ctx.extractor.collect(
        schema=OriginSchema,
        prompt="Cool. So where are you starting from? Which city?",
    )
    if not result.extracted.origin:
        return Interrupt(say="Which city will you be leaving from?", id="retry_origin")
    return Route("collect_destination")


async def collect_destination(state: TravelState, ctx: Context):                   # 7
    trip = state.trip_type or "domestic"
    result = await ctx.extractor.collect(
        schema=DestinationSchema,
        prompt=(
            f"And where do you want to go? "
            + ("Anywhere in India — mountains, beaches, cities, you name it."
               if trip == "domestic"
               else "We do trips all over — Southeast Asia, Europe, Middle East, wherever you're dreaming of.")
        ),
    )
    if not result.extracted.destination:
        return Interrupt(say="Where are you headed?", id="retry_dest")
    return Route("collect_departure_date")


async def collect_departure_date(state: TravelState, ctx: Context):                # 8
    result = await ctx.extractor.collect(
        fields=["departure_date"],
        prompt=(
            f"{state.origin} to {state.destination} — nice! "
            f"When are you looking to leave? Just give me a date, like 'April 15th' or even 'next Friday'."
        ),
    )
    if not result.extracted.departure_date:
        return Interrupt(say="What date works for you to start the trip?", id="retry_dep_date")
    return Route("collect_return_date")


async def collect_return_date(state: TravelState, ctx: Context):                   # 9
    result = await ctx.extractor.collect(
        fields=["return_date"],
        prompt=(
            f"And when are you coming back? Or if it's a one-way trip, just let me know."
        ),
    )
    if not result.extracted.return_date:
        return Interrupt(say="When do you want to head back? Or is it one way?", id="retry_ret_date")
    return Route("collect_num_travelers")


async def collect_num_travelers(state: TravelState, ctx: Context):                 # 10
    result = await ctx.extractor.collect(
        fields=["num_travelers"],
        prompt="How many people are going? Just you, or is it a group?",
    )
    if result.extracted.num_travelers is None:
        return Interrupt(say="How many of you are traveling?", id="retry_travelers")
    return Route("collect_budget")


async def collect_budget(state: TravelState, ctx: Context):                        # 11
    result = await ctx.extractor.collect(
        fields=["budget_tier"],
        prompt=(
            "Now, what kind of budget are we working with? "
            "Are you keeping it budget-friendly — like hostels and sleeper trains? "
            "Or more mid-range comfortable? Or going all out luxury — five stars, the works?"
        ),
    )
    if not result.extracted.budget_tier:
        return Interrupt(say="Just roughly — are we going budget, moderate, or luxury?", id="retry_budget")
    return Route("select_transport")


# ── 5C. TRANSPORT MODE SELECTION (node 12)

async def select_transport(state: TravelState, ctx: Context):                      # 12
    """Route to the correct transport branch."""

    mode = state.transport_mode
    if not mode:
        intent = await ctx.extractor.match_intent(
            intents={
                "train":  "train, railway, rail, rajdhani, shatabdi, express, IRCTC",
                "flight": "flight, fly, airplane, plane, airline, air, airport",
                "bus":    "bus, volvo, sleeper bus, road trip, KSRTC, RedBus",
            }
        )
        if intent in ("train", "flight", "bus"):
            mode = intent

    if mode == "train":
        return Route("search_trains", update={"transport_mode": "train"})
    if mode == "flight":
        return Route("search_flights", update={"transport_mode": "flight"})
    if mode == "bus":
        return Route("search_buses", update={"transport_mode": "bus"})
    return Interrupt(
        say=(
            f"Ask how they want to travel from {state.origin} to {state.destination}. "
            "Options are train, flight, or bus."
        ),
        id="retry_transport",
    )


# ── 5D. TRAIN BRANCH (nodes 13-17)

async def search_trains(state: TravelState, ctx: Context):                         # 13
    """Passthrough — just advance to select_train."""
    if state.train_name:
        return Route("train_class")
    return Route("select_train")


async def select_train(state: TravelState, ctx: Context):                          # 14
    if state.train_name:
        return Route("train_class")

    options = _format_trains(state.origin, state.destination, state.departure_date, state.num_travelers or 1)
    result = await ctx.extractor.collect(
        fields=["train_name"],
        prompt=(
            f"Read out these EXACT train options to the user — do NOT change, add, or make up any details:\n\n{options}\n\n"
            "Ask which one they want. If the user already mentioned a train name, pick it from the list above."
        ),
    )
    if not result.extracted.train_name:
        return Interrupt(say="Which train do you want — just say the name or the number?", id="retry_train")
    return Route("train_class")


async def train_class(state: TravelState, ctx: Context):                           # 15
    result = await ctx.extractor.collect(
        schema=TrainClassSchema,
        prompt=(
            f"{state.train_name} — solid pick. "
            f"Now what class do you want? Sleeper, 3AC, 2AC, 1AC, or Chair Car?"
        ),
    )
    if not result.extracted.train_class:
        return Interrupt(say="Sleeper, 3AC, 2AC, 1AC, or Chair Car — which one?", id="retry_train_class")
    return Route("train_berth")


async def train_berth(state: TravelState, ctx: Context):                           # 16
    if state.berth_preference:
        return Route("confirm_train")

    result = await ctx.extractor.collect(
        schema=BerthSchema,
        prompt="Any berth preference — lower, upper, middle, side? Or no preference?",
    )
    if result.extracted.berth_preference:
        return Route("confirm_train")

    return Interrupt(say="Lower, upper, middle, side — or no preference?", id="retry_berth")


async def confirm_train(state: TravelState, ctx: Context):                         # 17
    await ctx.say(
        f"Alright, so that's {state.train_name}, {state.train_class} class, "
        f"berth: {state.berth_preference}, {state.origin} to {state.destination} on {state.departure_date}, "
        f"{state.num_travelers} passenger(s). Seats are held — let's move on!"
    )
    return Route("transport_confirmed")


# ── 5E. FLIGHT BRANCH (nodes 18-23)

async def search_flights(state: TravelState, ctx: Context):                        # 18
    if state.flight_name:
        return Route("flight_class")
    return Route("select_flight")


async def select_flight(state: TravelState, ctx: Context):                         # 19
    if state.flight_name:
        return Route("flight_class")

    options = _format_flights(state.origin, state.destination, state.departure_date, state.num_travelers or 1)
    result = await ctx.extractor.collect(
        fields=["flight_name"],
        prompt=(
            f"Read out these EXACT flight options to the user — do NOT change, add, or make up any details:\n\n{options}\n\n"
            "Ask which one they want. If the user already mentioned a flight, pick it from the list above."
        ),
    )
    if not result.extracted.flight_name:
        return Interrupt(say="Which flight do you want — just say the airline or number?", id="retry_flight")
    return Route("flight_class")


async def flight_class(state: TravelState, ctx: Context):                          # 20
    result = await ctx.extractor.collect(
        schema=FlightClassSchema,
        prompt=(
            f"Nice, {state.flight_name} it is. "
            f"Are you going economy, premium economy, business, or splurging on first class?"
        ),
    )
    if not result.extracted.flight_class:
        return Interrupt(say="Economy, business, or something else?", id="retry_fl_class")
    return Route("flight_seat")


async def flight_seat(state: TravelState, ctx: Context):                           # 21
    if state.seat_preference:
        return Route("flight_meal")

    result = await ctx.extractor.collect(
        schema=SeatSchema,
        prompt="Do you have a seat preference — window, aisle, or doesn't matter?",
    )
    if result.extracted.seat_preference:
        return Route("flight_meal")

    return Interrupt(say="Window, aisle, or no preference?", id="retry_seat")


async def flight_meal(state: TravelState, ctx: Context):                           # 22
    if state.meal_preference:
        return Route("confirm_flight")

    result = await ctx.extractor.collect(
        schema=MealSchema,
        prompt="Want to add a meal? Veg, non-veg, vegan, jain — or say skip?",
    )
    if result.extracted.meal_preference:
        return Route("confirm_flight")

    return Interrupt(say="Any meal — veg, non-veg, vegan, jain? Or skip?", id="retry_meal")


async def confirm_flight(state: TravelState, ctx: Context):                        # 23
    await ctx.say(
        f"Okay so that's {state.flight_name}, {state.flight_class}, "
        f"{state.seat_preference} seat, meal: {state.meal_preference}, "
        f"{state.origin} to {state.destination} on {state.departure_date}, "
        f"{state.num_travelers} passenger(s). Seats are held — moving on!"
    )
    return Route("transport_confirmed")


# ── 5F. BUS BRANCH (nodes 24-27)

async def search_buses(state: TravelState, ctx: Context):                          # 24
    if state.bus_name:
        return Route("bus_seat")
    return Route("select_bus")


async def select_bus(state: TravelState, ctx: Context):                            # 25
    if state.bus_name:
        return Route("bus_seat")

    options = _format_buses(state.origin, state.destination, state.departure_date)
    result = await ctx.extractor.collect(
        fields=["bus_name"],
        prompt=(
            f"Read out these EXACT bus options to the user — do NOT change, add, or make up any details:\n\n{options}\n\n"
            "Ask which one they want. If the user already mentioned a bus, pick it from the list above."
        ),
    )
    if not result.extracted.bus_name:
        return Interrupt(say="Which bus works for you — just say the operator or number?", id="retry_bus")
    return Route("bus_seat")


async def bus_seat(state: TravelState, ctx: Context):                              # 26
    if state.bus_seat_type:
        return Route("confirm_bus")

    result = await ctx.extractor.collect(
        schema=BusSeatSchema,
        prompt="Do you want a regular seat, sleeper, or semi-sleeper?",
    )
    if result.extracted.bus_seat_type:
        return Route("confirm_bus")

    return Interrupt(say="Seater, sleeper, or semi-sleeper?", id="retry_bus_seat")


async def confirm_bus(state: TravelState, ctx: Context):                           # 27
    await ctx.say(
        f"So that's {state.bus_name}, {state.bus_seat_type}, "
        f"{state.origin} to {state.destination} on {state.departure_date} "
        f"for {state.num_travelers}. Seats are held — let's keep going!"
    )
    return Route("transport_confirmed")


# ── 5G. TRANSPORT MERGE (node 28)

async def transport_confirmed(state: TravelState, ctx: Context):                   # 28
    """Merge point after any transport branch. Advance to hotel."""
    ctx.update_states({"transport_confirmed": True})
    return Route("ask_hotel")


# ── 5H. HOTEL BOOKING (nodes 29-37)

async def ask_hotel(state: TravelState, ctx: Context):                             # 29
    """Ask if user needs hotel booking."""
    # Check if LLM already resolved this
    if state.needs_hotel is True:
        return Route("collect_hotel_location")
    if state.needs_hotel is False:
        return Route("collect_interests")

    intent = await ctx.extractor.match_intent(
        intents={
            "yes":  "yes, need hotel, book hotel, want hotel, accommodation, stay, lodge, sure, please",
            "no":   "no, don't need, already booked, staying with friends, family, no hotel, skip, nah",
        }
    )
    if intent == "yes":
        return Route("collect_hotel_location", update={"needs_hotel": True})
    if intent == "no":
        return Route("collect_interests", update={"needs_hotel": False})
    return Interrupt(
        say=(
            f"Ask if they need a hotel in {state.destination}. "
            "We can find hotels, homestays, whatever works."
        ),
        id="retry_hotel_ask",
    )


async def collect_hotel_location(state: TravelState, ctx: Context):               # 30
    # Skip if already filled or if later hotel fields are done (out-of-order extraction)
    if state.hotel_location:
        return Route("collect_checkin_checkout")
    if state.hotel_name:
        ctx.update_states({"hotel_location": "city center"})
        return Route("collect_checkin_checkout")

    result = await ctx.extractor.collect(
        fields=["hotel_location"],
        prompt=(
            f"Any preference on where to stay in {state.destination}? "
            f"Like near the main market, close to the tourist area, or near the station or airport?"
        ),
    )
    if result.extracted.hotel_location:
        return Route("collect_checkin_checkout")
    return Interrupt(say="Which part of town do you want to be in?", id="retry_hotel_loc")


async def collect_checkin_checkout(state: TravelState, ctx: Context):              # 31
    if state.checkin_date and state.checkout_date:
        return Route("collect_room_type")

    # Default to trip dates — no need to ask, just confirm
    ctx.update_states({
        "checkin_date": state.checkin_date or state.departure_date,
        "checkout_date": state.checkout_date or state.return_date,
    })
    return Route("collect_room_type")


async def collect_room_type(state: TravelState, ctx: Context):                    # 32
    if state.room_type:
        return Route("collect_num_rooms")

    rec = {"budget": "a single or dormitory", "moderate": "a double or twin",
           "luxury": "a suite or a nice double"}.get(state.budget_tier or "moderate", "a double")
    result = await ctx.extractor.collect(
        schema=RoomTypeSchema,
        prompt=(
            f"What kind of room? Single, double, twin, suite, or dormitory? "
            f"Since you're going {state.budget_tier}, I'd say {rec} would be a good fit."
        ),
    )
    if not result.extracted.room_type:
        return Interrupt(say="Single, double, twin, suite, or dorm?", id="retry_room")
    return Route("collect_num_rooms")


async def collect_num_rooms(state: TravelState, ctx: Context):                    # 33
    if state.num_rooms:
        return Route("search_hotels")

    result = await ctx.extractor.collect(
        fields=["num_rooms"],
        prompt=f"And how many rooms? You've got {state.num_travelers} people going.",
    )
    if result.extracted.num_rooms is None:
        return Interrupt(say="How many rooms do you need?", id="retry_rooms")
    return Route("search_hotels")


async def search_hotels(state: TravelState, ctx: Context):                        # 34
    if state.hotel_name:
        return Route("hotel_extras")
    return Route("select_hotel")


async def select_hotel(state: TravelState, ctx: Context):                         # 35
    if state.hotel_name:
        return Route("hotel_extras")

    options = _format_hotels(
        state.destination, state.budget_tier or "moderate",
        state.hotel_location or "city center", state.num_rooms or 1, state.room_type or "double",
    )
    result = await ctx.extractor.collect(
        fields=["hotel_name"],
        prompt=(
            f"Read out these EXACT hotel options to the user — do NOT change, add, or make up any details:\n\n{options}\n\n"
            "Ask which one they like. If the user already mentioned a hotel, pick it from the list above."
        ),
    )
    if not result.extracted.hotel_name:
        return Interrupt(say="Which hotel catches your eye?", id="retry_hotel")
    return Route("hotel_extras")


async def hotel_extras(state: TravelState, ctx: Context):                          # 36
    result = await ctx.extractor.collect(
        fields=["hotel_extras"],
        prompt=(
            "Anything extra you want with the hotel? Like breakfast, "
            "airport or station pickup, spa, early check-in, late checkout? "
            "Or you're good without any of that?"
        ),
    )
    if not result.extracted.hotel_extras:
        ctx.update_states({"hotel_extras": "none"})
    return Route("confirm_hotel")


async def confirm_hotel(state: TravelState, ctx: Context):                         # 37
    await ctx.say(
        f"Quick recap — {state.hotel_name} near {state.hotel_location}, "
        f"{state.num_rooms} {state.room_type} room(s), "
        f"{state.checkin_date} to {state.checkout_date}, extras: {state.hotel_extras}. "
        f"Rooms are held!"
    )
    return Route("collect_interests")


# ── 5I. ITINERARY PLANNING (nodes 38-40) ─

async def collect_interests(state: TravelState, ctx: Context):                     # 38
    result = await ctx.extractor.collect(
        fields=["interests"],
        prompt=(
            f"Now the fun part — what do you actually want to do in {state.destination}? "
            f"Like are you into food and street food, sightseeing, adventure stuff, "
            f"history, shopping, nightlife, trekking, temples, photography — "
            f"tell me whatever you're into and I'll build the itinerary around that."
        ),
    )
    if not result.extracted.interests:
        return Interrupt(say=f"What kind of stuff are you into? That'll help me plan your days in {state.destination}.", id="retry_interests")
    return Route("generate_itinerary")


async def generate_itinerary(state: TravelState, ctx: Context):                   # 39
    await ctx.say("Let me put together a day-by-day plan for you...")
    return Route("review_itinerary")


async def review_itinerary(state: TravelState, ctx: Context):                     # 40
    """LLM generates itinerary, user approves or requests changes."""
    if state.itinerary_approved:
        if state.trip_type == "international":
            return Route("check_visa")
        return Route("offer_insurance")

    result = await ctx.extractor.collect(
        fields=["itinerary_approved"],
        prompt=(
            f"Build a day-by-day plan for {state.customer_name} in {state.destination}, "
            f"{state.departure_date} to {state.return_date}. "
            f"{state.num_travelers} people, {state.budget_tier} budget. "
            f"They're into: {state.interests}. "
            f"Getting there by {state.transport_mode} ({state.train_name or state.flight_name or state.bus_name}). "
            f"Staying at {state.hotel_name or 'not booked yet'} in {state.hotel_location or 'TBD'}. "
            f"Plan out morning, afternoon, and evening for each day. "
            f"Throw in restaurant suggestions, rough costs, and local tips. "
            f"Keep it conversational — like a friend giving travel advice. "
            f"Then ask if they want to tweak anything or if it looks good."
        ),
    )

    if result.extracted.itinerary_approved is True:
        if state.trip_type == "international":
            return Route("check_visa")
        return Route("offer_insurance")

    if result.extracted.itinerary_approved is False:
        ctx.update_states({"itinerary_approved": None})
        return Interrupt(
            say=(
                "Sure, what do you want to change? I can swap things around, "
                "add something new, drop something you don't like — just tell me."
            ),
            id="retry_itinerary",
        )

    return Interrupt(
        prompt="So what do you think — looks good, or want me to change something?",
        id="retry_itinerary_confirm",
    )


# ── 5J. VISA & PASSPORT — international only (nodes 41-43) 

async def check_visa(state: TravelState, ctx: Context):                            # 41
    """LLM checks visa requirements, collects visa status."""
    if state.visa_status:
        if state.visa_status in ("have_visa", "not_needed"):
            return Route("offer_insurance")
        return Route("collect_passport")

    result = await ctx.extractor.collect(
        fields=["visa_status"],
        prompt=(
            f"Since this is an international trip to {state.destination}, check the visa situation "
            f"for Indian passport holders. Let them know if they need a visa, if there's visa on arrival, "
            f"or if it's visa-free. If they need one, mention how long it takes and what docs they'll need. "
            f"Then ask: do they already have a visa, need help getting one, or is it not required?"
        ),
    )
    if not result.extracted.visa_status:
        return Interrupt(say="Do you already have a visa, need help getting one, or is it not needed for this trip?", id="retry_visa")
    if result.extracted.visa_status in ("have_visa", "not_needed"):
        return Route("offer_insurance")
    return Route("collect_passport")


async def collect_passport(state: TravelState, ctx: Context):                      # 42
    result = await ctx.extractor.collect(
        fields=["passport_number", "passport_expiry"],
        prompt=(
            "I'll need your passport details for the booking — just the passport number "
            "and when it expires. Quick heads up, it needs to be valid for at least 6 months "
            "from your travel date."
        ),
    )
    if not result.extracted.passport_number:
        return Interrupt(say="What's your passport number?", id="retry_passport")
    if not result.extracted.passport_expiry:
        return Interrupt(say="And when does it expire?", id="retry_passport_exp")
    return Route("visa_info")


async def visa_info(state: TravelState, ctx: Context):                             # 43
    """Provide visa guidance and move on."""
    await ctx.say(
        f"No worries, we can help with the visa for {state.destination}. "
        f"Our team will reach out to you at {state.phone} with all the application details "
        f"and next steps. For now, let's continue with the rest of the booking."
    )
    return Route("offer_insurance")


# ── 5K. TRAVEL INSURANCE (nodes 44-45) ───

async def offer_insurance(state: TravelState, ctx: Context):                       # 44
    if state.wants_insurance is True:
        return Route("select_insurance")
    if state.wants_insurance is False:
        return Route("order_summary")

    intent = await ctx.extractor.match_intent(
        intents={
            "yes": "yes, want insurance, sure, get insurance, add insurance, interested, okay, why not",
            "no":  "no, skip, don't need, not interested, no insurance, pass, nah, no thanks",
        }
    )
    if intent == "yes":
        return Route("select_insurance", update={"wants_insurance": True})
    if intent == "no":
        return Route("order_summary", update={"wants_insurance": False})
    trip = "international" if state.trip_type == "international" else "domestic"
    extra = "Since this is an international trip, strongly recommend it. " if trip == "international" else ""
    return Interrupt(
        say=(
            f"Ask if they want travel insurance. {extra}"
            "It covers cancellations, medical emergencies, lost bags, and delays. "
            "Plans start at 500 rupees."
        ),
        id="retry_insurance_ask",
    )


async def select_insurance(state: TravelState, ctx: Context):                      # 45
    result = await ctx.extractor.collect(
        schema=InsurancePlanSchema,
        prompt=(
            "We've got three plans — "
            "Basic at 500 rupees, covers cancellations and delays. "
            "Standard at 1200, adds medical up to 5 lakhs. "
            "And Premium at 2500 — that's the full package, medical up to 15 lakhs, gadgets, adventure sports. "
            "Which one?"
        ),
    )
    if not result.extracted.insurance_plan:
        return Interrupt(say="Basic, standard, or premium — what works?", id="retry_insurance")
    return Route("order_summary")


# ── 5L. PAYMENT (nodes 46-49)

async def order_summary(state: TravelState, ctx: Context):                         # 46
    """Calculate total and present full booking summary via TTS."""
    total = _estimate_cost(state)
    ctx.update_states({"total_amount": total, "booking_ref": _gen_ref()})

    transport_detail = state.train_name or state.flight_name or state.bus_name or "N/A"
    hotel_detail = f"{state.hotel_name}, {state.num_rooms} {state.room_type} room(s)" if state.needs_hotel else "no hotel"
    insurance_detail = f"{(state.insurance_plan or 'none')} plan" if state.wants_insurance else "no insurance"

    await ctx.say(
        f"Alright {state.customer_name}, here's your full booking: "
        f"{state.origin} to {state.destination}, {state.trip_type} trip, "
        f"{state.departure_date} to {state.return_date}, {state.num_travelers} traveler(s). "
        f"Transport: {state.transport_mode}, {transport_detail}. "
        f"Hotel: {hotel_detail}. Insurance: {insurance_detail}. "
        f"Total comes to about {total} rupees. Your booking ref is {state.booking_ref}."
    )
    return Route("select_payment")


async def select_payment(state: TravelState, ctx: Context):                        # 47
    result = await ctx.extractor.collect(
        schema=PaymentMethodSchema,
        prompt="How do you want to pay? UPI, card, net banking, or wallet?",
    )
    if not result.extracted.payment_method:
        return Interrupt(say="UPI, card, net banking, or wallet?", id="retry_pay")
    return Route("initiate_payment")


async def initiate_payment(state: TravelState, ctx: Context):                      # 48
    """Create payment link and wait for webhook callback."""
    if ctx.human_input is None:
        try:
            link = await _create_payment_link(
                ref=state.booking_ref or _gen_ref(),
                amount=state.total_amount or 0,
                name=state.customer_name or "Traveler",
                phone=state.phone or "",
            )
        except Exception as exc:
            logger.error("[PAYMENT] Link creation failed: %s", exc)
            return Interrupt(say="Hmm, having a bit of trouble with the payment link. Give me a sec.", id="retry_pay_link")

        ctx.update_states({"payment_link": link, "payment_status": "pending"})
        return HumanInLoop(
            reason="awaiting_payment",
            say=(
                f"Done — I've sent a payment link to your phone ending in ...{(state.phone or '0000')[-4:]}. "
                f"It's Rs {state.total_amount:,}. Just complete it in the next {PAYMENT_LINK_EXPIRE_MIN} minutes "
                f"and we're all set."
            ),
            timeout=float(PAYMENT_LINK_EXPIRE_MIN * 60),
        )

    # Resumed by webhook
    status = ctx.human_input.get("payment_status", "failed")
    pay_id = ctx.human_input.get("payment_id", "")
    ctx.update_states({"payment_status": status, "payment_id": pay_id})

    if status == "paid":
        return Route("payment_success")
    return Route("payment_failed")


async def payment_success(state: TravelState, ctx: Context):                       # 49
    """Payment received — proceed to confirmation."""
    return Route("send_confirmation")


async def payment_failed(state: TravelState, ctx: Context):                        # 49b
    await ctx.say(
        f"Ah, looks like the payment didn't go through. "
        f"Don't worry, your booking ref {state.booking_ref} is saved — "
        f"you can try again or just give us a call at 1800-YATRA-00 and we'll sort it out."
    )
    return END


# ── 5M. CONFIRMATION & FAREWELL (nodes 50-53) 

async def send_confirmation(state: TravelState, ctx: Context):                     # 50
    """Send SMS and email confirmation."""
    transport_detail = state.train_name or state.flight_name or state.bus_name or "N/A"

    sms_body = (
        f"Yatra Express Booking Confirmed! Ref: {state.booking_ref} | "
        f"{state.customer_name} | {state.origin} to {state.destination} | "
        f"{state.departure_date} to {state.return_date} | "
        f"{state.transport_mode}: {transport_detail} | "
        f"Amount: Rs {state.total_amount:,} | PayID: {state.payment_id}"
    )
    await _send_sms(state.phone or "", sms_body)

    await ctx.say(
        f"It's all booked, {state.customer_name}! "
        f"Ref {state.booking_ref} — {state.origin} to {state.destination}, "
        f"{state.departure_date} to {state.return_date}. "
        f"Confirmation text is on its way to your phone, and your e-tickets "
        f"plus the full itinerary will be in your inbox at {state.email} shortly."
    )
    return Route("collect_feedback")


async def collect_feedback(state: TravelState, ctx: Context):                      # 51
    result = await ctx.extractor.collect(
        schema=FeedbackSchema,
        prompt=(
            "Before you go — how was the booking experience? "
            "Give me a quick rating, 1 to 5. And feel free to tell me if anything could be better."
        ),
    )
    if result.extracted.feedback_rating is None:
        intent = await ctx.extractor.match_intent(
            intents={
                "skip": "skip, no, later, not now, pass, bye",
                "rate": "1, 2, 3, 4, 5, stars, great, good, bad, okay",
            }
        )
        if intent == "skip":
            return Route("farewell")
        return Interrupt(say="Just a quick 1 to 5 would be great. Or say skip, totally fine.", id="retry_feedback")

    rating = result.extracted.feedback_rating
    msg = {1: "Oh no, sorry about that", 2: "Noted, we'll work on it", 3: "Fair enough, appreciate it",
           4: "Glad it went well", 5: "That's great to hear"}.get(rating, "Thanks")
    await ctx.say(f"{msg}! Noted that down.")
    return Route("farewell")


async def farewell(state: TravelState, ctx: Context):                              # 52
    await ctx.say(
        f"You're all set, {state.customer_name}! Booking ref is {state.booking_ref} — "
        f"save that just in case. If you need to change anything later, "
        f"just call 1800-YATRA-00 or message us right here. "
        f"Have an amazing time in {state.destination}! Take care!"
    )
    return END


#   6. GRAPH WIRING  (51 nodes, ~15 transitions)                                


# Register all nodes
# Customer onboarding
graph.add_node("welcome",                welcome)
graph.add_node("collect_name",           collect_name)
graph.add_node("collect_phone",          collect_phone)
graph.add_node("collect_email",          collect_email)

# Trip planning
graph.add_node("detect_trip_type",       detect_trip_type)
graph.add_node("collect_origin",         collect_origin)
graph.add_node("collect_destination",    collect_destination)
graph.add_node("collect_departure_date", collect_departure_date)
graph.add_node("collect_return_date",    collect_return_date)
graph.add_node("collect_num_travelers",  collect_num_travelers)
graph.add_node("collect_budget",         collect_budget)

# Transport selection
graph.add_node("select_transport",       select_transport)

# Train branch
graph.add_node("search_trains",          search_trains)
graph.add_node("select_train",           select_train)
graph.add_node("train_class",            train_class)
graph.add_node("train_berth",            train_berth)
graph.add_node("confirm_train",          confirm_train)

# Flight branch
graph.add_node("search_flights",         search_flights)
graph.add_node("select_flight",          select_flight)
graph.add_node("flight_class",           flight_class)
graph.add_node("flight_seat",            flight_seat)
graph.add_node("flight_meal",            flight_meal)
graph.add_node("confirm_flight",         confirm_flight)

# Bus branch
graph.add_node("search_buses",           search_buses)
graph.add_node("select_bus",             select_bus)
graph.add_node("bus_seat",               bus_seat)
graph.add_node("confirm_bus",            confirm_bus)

# Transport merge
graph.add_node("transport_confirmed",    transport_confirmed)

# Hotel booking
graph.add_node("ask_hotel",              ask_hotel)
graph.add_node("collect_hotel_location", collect_hotel_location)
graph.add_node("collect_checkin_checkout", collect_checkin_checkout)
graph.add_node("collect_room_type",      collect_room_type)
graph.add_node("collect_num_rooms",      collect_num_rooms)
graph.add_node("search_hotels",          search_hotels)
graph.add_node("select_hotel",           select_hotel)
graph.add_node("hotel_extras",           hotel_extras)
graph.add_node("confirm_hotel",          confirm_hotel)

# Itinerary
graph.add_node("collect_interests",      collect_interests)
graph.add_node("generate_itinerary",     generate_itinerary)
graph.add_node("review_itinerary",       review_itinerary)

# Visa & Passport (international)
graph.add_node("check_visa",             check_visa)
graph.add_node("collect_passport",       collect_passport)
graph.add_node("visa_info",              visa_info)

# Insurance
graph.add_node("offer_insurance",        offer_insurance)
graph.add_node("select_insurance",       select_insurance)

# Payment
graph.add_node("order_summary",          order_summary)
graph.add_node("select_payment",         select_payment)
graph.add_node("initiate_payment",       initiate_payment)
graph.add_node("payment_success",        payment_success)
graph.add_node("payment_failed",         payment_failed)

# Confirmation & farewell
graph.add_node("send_confirmation",      send_confirmation)
graph.add_node("collect_feedback",       collect_feedback)
graph.add_node("farewell",               farewell)

# ── Wire transitions 

# onboarding → trip planning → transport selection
graph.add_transition(
    START,
    "welcome",
    "collect_name",
    "collect_phone",
    "collect_email",
    "detect_trip_type",
    "collect_origin",
    "collect_destination",
    "collect_departure_date",
    "collect_return_date",
    "collect_num_travelers",
    "collect_budget",
    "select_transport",
)

# Train branch: search → select → class → berth → confirm → merge
graph.add_transition(
    "search_trains",
    "select_train",
    "train_class",
    "train_berth",
    "confirm_train",
    "transport_confirmed",
)

# Flight branch: search → select → class → seat → meal → confirm → merge
graph.add_transition(
    "search_flights",
    "select_flight",
    "flight_class",
    "flight_seat",
    "flight_meal",
    "confirm_flight",
    "transport_confirmed",
)

# Bus branch: search → select → seat → confirm → merge
graph.add_transition(
    "search_buses",
    "select_bus",
    "bus_seat",
    "confirm_bus",
    "transport_confirmed",
)

# After transport merge → hotel decision
graph.add_transition(
    "transport_confirmed",
    "ask_hotel",
)

# Hotel full branch
graph.add_transition(
    "collect_hotel_location",
    "collect_checkin_checkout",
    "collect_room_type",
    "collect_num_rooms",
    "search_hotels",
    "select_hotel",
    "hotel_extras",
    "confirm_hotel",
    "collect_interests",
)

# Itinerary
graph.add_transition(
    "collect_interests",
    "generate_itinerary",
    "review_itinerary",
)

# Visa flow (international)
graph.add_transition(
    "check_visa",
    "collect_passport",
    "visa_info",
    "offer_insurance",
)

# Insurance → payment
graph.add_transition(
    "select_insurance",
    "order_summary",
)

# Payment flow
graph.add_transition(
    "order_summary",
    "select_payment",
    "initiate_payment",
)

# Payment success → confirmation → feedback → farewell → END
graph.add_transition(
    "payment_success",
    "send_confirmation",
    "collect_feedback",
    "farewell",
    END,
)

# Payment failed → END
graph.add_transition("payment_failed", END)


logger.info("[GRAPH] Travel Agency Topology:\n%s", graph.get_graph_status())


#   7. PAYMENT WEBHOOK SERVER                                                    

async def _start_payment_webhook() -> None:
    try:
        from aiohttp import web

        def _verify_sig(body: bytes, sig: str) -> bool:
            import hmac, hashlib
            expected = hmac.new(RAZORPAY_WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
            return hmac.compare_digest(expected, sig)

        async def razorpay_webhook(request: web.Request) -> web.Response:
            body = await request.read()
            if RAZORPAY_WEBHOOK_SECRET:
                sig = request.headers.get("X-Razorpay-Signature", "")
                if not _verify_sig(body, sig):
                    return web.json_response({"error": "invalid signature"}, status=400)
            try:
                event = __import__("json").loads(body)
            except Exception:
                return web.json_response({"error": "invalid JSON"}, status=400)
            event_type = event.get("event", "")
            if event_type == "payment_link.paid":
                entity = event.get("payload", {}).get("payment", {}).get("entity", {})
                resume_payload = {"payment_status": "paid", "payment_id": entity.get("id", "")}
            elif event_type in ("payment_link.expired", "payment_link.cancelled"):
                resume_payload = {"payment_status": "failed", "payment_id": ""}
            else:
                return web.json_response({"status": "ignored"})
            if not graph._is_paused:
                return web.json_response({"error": "not awaiting payment"}, status=409)
            await graph.resume_with_human_input(resume_payload)
            return web.json_response({"status": "ok"})

        async def mock_pay_page(request: web.Request) -> web.Response:
            ref = request.query.get("ref", "")
            amount = request.query.get("amount", "0")
            html = f"""<!DOCTYPE html>
<html><head><title>Mock Payment</title>
<style>body{{font-family:sans-serif;max-width:400px;margin:60px auto;text-align:center}}
button{{padding:14px 32px;margin:10px;font-size:16px;border:none;border-radius:6px;cursor:pointer}}
.pay{{background:#2ecc71;color:#fff}}.fail{{background:#e74c3c;color:#fff}}</style></head>
<body><h2>Yatra Express - Mock Payment</h2>
<p>Booking: <b>{ref}</b> | Amount: <b>Rs {amount}</b></p>
<form method="POST" action="/mock-pay">
<input type="hidden" name="ref" value="{ref}">
<button class="pay" name="action" value="pay" type="submit">Pay Now</button>
<button class="fail" name="action" value="fail" type="submit">Simulate Failure</button>
</form></body></html>"""
            return web.Response(text=html, content_type="text/html")

        async def mock_pay_submit(request: web.Request) -> web.Response:
            data = await request.post()
            ref = data.get("ref", "")
            action = data.get("action", "fail")
            resume_payload = (
                {"payment_status": "paid", "payment_id": f"mock_{ref}"}
                if action == "pay"
                else {"payment_status": "failed", "payment_id": ""}
            )
            if not graph._is_paused:
                return web.Response(text="Not waiting for payment.", status=409)
            await graph.resume_with_human_input(resume_payload)
            status = "Payment successful!" if action == "pay" else "Payment failed."
            return web.Response(text=f"<h2>{status}</h2><p>Close this tab.</p>", content_type="text/html")

        app = web.Application()
        app.router.add_post("/razorpay-webhook", razorpay_webhook)
        app.router.add_get("/mock-pay", mock_pay_page)
        app.router.add_post("/mock-pay", mock_pay_submit)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", WEBHOOK_PORT)
        await site.start()
        logger.info("[WEBHOOK] Payment server listening on :%d", WEBHOOK_PORT)
    except ImportError:
        logger.warning("[WEBHOOK] aiohttp not installed — webhook server skipped")


#   8. AGENT + PIPELINE                                                          

class TravelAgent(Agent):

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a travel agent at Yatra Express named MAYA. Talk like a real person — friendly, "
                "helpful, not robotic. Use casual language, contractions, and short sentences. "
                "Don't list things in bullet points unless the user asks — just talk normally. "
                "When showing train/flight/hotel options, present ONLY the exact options "
                "provided in the prompt — do NOT invent, modify, or add any names, numbers, "
                "timings, or prices. Read them out naturally, like you're telling a friend "
                "what's on your screen. You may add a brief travel tip occasionally "
                "but don't overdo it.\n\n"
            ),
        )

    async def on_enter(self) -> None:
        logger.info("[AGENT] Travel agency session started")
        await self.session.say(
            "Hey, welcome to Yatra Express! "
            "I'll help you sort out your whole trip — tickets, hotel, itinerary, the works. "
            "Let's get started."
        )

    async def on_exit(self) -> None:
        logger.info("[AGENT] Travel agency session ended")


async def entrypoint(ctx: JobContext) -> None:
    await _start_payment_webhook()

    agent = TravelAgent()

    pipeline = Pipeline(
        stt=DeepgramSTT(model="nova-2", language="en-IN"),
        llm=GoogleLLM(model="gemini-2.5-flash"),
        tts=CartesiaTTS(model="sonic-3"),
        conversational_graph=graph,
    )

    session = AgentSession(agent=agent, pipeline=pipeline)
    await graph.compile(user_config={"user_id":"test_user"})
    await graph.resume()
    graph.get_graph_status()
    graph.visualize()
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    return JobContext(room_options=RoomOptions(
        name="Yatra Express Travel Agent",
        playground=True,
    ))


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
