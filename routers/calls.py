import logging
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from twilio.rest import Client
from xml.sax.saxutils import quoteattr

from config import NGROK_URL, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER
from schemas import CallRequest, ChatTestRequest
from services import agent_manager, claude_client, conversation_store
from services.call_logger import get_call_log, CALL_LOGS_DIR

logger = logging.getLogger("calls")
router = APIRouter(tags=["calls"])


@router.post("/make-call")
def make_call(request: CallRequest):
    """Initiate an outbound call to the specified phone number using the given agent."""
    if not agent_manager.get_agent(request.agent_id):
        raise HTTPException(status_code=404, detail=f"Agent '{request.agent_id}' not found")

    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_PHONE_NUMBER:
        raise HTTPException(status_code=500, detail="Twilio is not configured. Set TWILIO_* vars in .env.")
    if not NGROK_URL:
        raise HTTPException(status_code=500, detail="NGROK_URL is not configured in .env.")

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            url=f"{NGROK_URL}/outgoing-call?agent_id={request.agent_id}",
            to=request.to_phone_number,
            from_=TWILIO_PHONE_NUMBER,
            record=True,
            recording_status_callback=f"{NGROK_URL}/recording-status",
            recording_status_callback_method="POST",
        )
        return {"call_sid": call.sid, "status": "success", "agent_id": request.agent_id}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@router.api_route("/outgoing-call", methods=["GET", "POST"])
async def handle_outgoing_call(request: Request):
    """Twilio webhook: returns TwiML that connects the call to our ConversationRelay
    WebSocket. ConversationRelay handles speech-to-text and text-to-speech for us -
    our server only ever sees/produces plain text."""
    agent_id = request.query_params.get("agent_id", "default")
    agent = agent_manager.get_agent(agent_id) or agent_manager.get_agent("default")

    ws_url = f"wss://{request.url.hostname}/media-stream"
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response><Connect>"
        f"<ConversationRelay url={quoteattr(ws_url)} "
        f"welcomeGreeting={quoteattr(agent['welcome_greeting'])} "
        f"voice={quoteattr(agent['voice'])} "
        'ttsProvider="Google" interruptible="true">'
        f"<Parameter name=\"agentId\" value={quoteattr(agent['id'])}/>"
        "</ConversationRelay>"
        "</Connect></Response>"
    )
    return HTMLResponse(content=twiml, media_type="application/xml")


@router.api_route("/recording-status", methods=["POST"])
async def handle_recording_status(request: Request):
    """Handle recording status updates from Twilio."""
    form_data = await request.form()
    print("Recording status update:", dict(form_data))
    return {"status": "received"}


@router.get("/calls")
def list_calls():
    """List recent calls (real phone calls and /chat/test sessions alike)."""
    return conversation_store.list_calls()


@router.get("/calls/{call_sid}")
def get_call_summary(call_sid: str):
    """Full transcript + sentiment timeline + escalation flag for one call."""
    summary = conversation_store.get_call_summary(call_sid)
    if not summary:
        raise HTTPException(status_code=404, detail="Call not found")
    # Attach file-based call log if available.
    call_log = get_call_log(call_sid)
    if call_log:
        summary["call_log"] = call_log
    return summary


@router.get("/call-logs")
def list_call_logs(limit: int = 50):
    """List available call log files (newest first)."""
    if not os.path.isdir(CALL_LOGS_DIR):
        return []
    files = sorted(
        (f for f in os.listdir(CALL_LOGS_DIR) if f.endswith(".json")),
        key=lambda f: os.path.getmtime(os.path.join(CALL_LOGS_DIR, f)),
        reverse=True,
    )
    results = []
    import json
    for fname in files[:limit]:
        try:
            with open(os.path.join(CALL_LOGS_DIR, fname), "r", encoding="utf-8") as fh:
                data = json.load(fh)
            results.append({
                "call_sid": data.get("call_sid"),
                "scenario_id": data.get("scenario_id"),
                "language": data.get("language"),
                "status": data.get("status"),
                "outcome": data.get("outcome"),
                "started_at": data.get("started_at"),
                "ended_at": data.get("ended_at"),
                "total_turns": len(data.get("turns", [])),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return results


@router.get("/call-logs/{call_sid}")
def get_call_log_detail(call_sid: str):
    """Full per-call transcript log file."""
    log = get_call_log(call_sid)
    if not log:
        raise HTTPException(status_code=404, detail="Call log not found")
    return log


@router.post("/chat/test")
async def chat_test(payload: ChatTestRequest):
    """Talk to an agent over plain HTTP - no Twilio/ngrok/phone needed.

    Lets you validate an agent's prompt, knowledge base, and sentiment-aware
    behavior immediately after setting an ANTHROPIC_API_KEY, before wiring up
    any telephony at all. `session_id` is any string you pick to keep one
    back-and-forth conversation together across calls to this endpoint.
    """
    agent = agent_manager.get_agent(payload.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{payload.agent_id}' not found")

    call_sid = f"test:{payload.session_id}"
    if not conversation_store.get_call(call_sid):
        conversation_store.create_call(call_sid, agent["id"], "test-user", agent["id"], "test")

    try:
        sentiment = await claude_client.classify_sentiment(payload.message)
        conversation_store.add_message(
            call_sid,
            "user",
            payload.message,
            sentiment_label=sentiment["sentiment"],
            sentiment_score=sentiment["score"],
            emotion=sentiment["emotion"],
            escalate=sentiment["escalate"],
        )

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in conversation_store.get_messages(call_sid)
        ]
        if sentiment["sentiment"] == "negative" or sentiment["escalate"]:
            history[-1]["content"] = (
                f"[Caller tone: {sentiment['emotion']}, sentiment={sentiment['sentiment']}] "
                f"{payload.message}"
            )

        system_prompt = agent_manager.build_system_prompt(agent)
        reply = await claude_client.generate_reply(
            system_prompt, history, agent["model"], agent["effort"]
        )
        conversation_store.add_message(call_sid, "assistant", reply)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "call_sid": call_sid,
        "reply": reply,
        "sentiment": sentiment,
    }
