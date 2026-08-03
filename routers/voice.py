import json
import logging

from fastapi import APIRouter, WebSocket
from fastapi.websockets import WebSocketDisconnect

from services import agent_manager, claude_client, conversation_store
from services.call_logger import init_call_log, log_turn, finalize_call_log

logger = logging.getLogger("voice")
router = APIRouter()


@router.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Twilio ConversationRelay speaks plain text to us over this socket:

    Twilio -> us:  {"type": "setup", "callSid": ..., "customParameters": {...}}
                   {"type": "prompt", "voicePrompt": "<caller's transcribed speech>", "last": true}
                   {"type": "interrupt", ...}
    us -> Twilio:  {"type": "text", "token": "<partial or full reply>", "last": bool}

    This is the two-way loop: every caller utterance is transcribed by Twilio,
    handed to us as text, run through sentiment analysis + Claude with the
    full conversation history, and the reply is streamed back token-by-token
    for Twilio to speak. Note: ConversationRelay is a Twilio public-beta
    product - double check exact message field names against Twilio's current
    docs if this stops working after a Twilio-side update.
    """
    await websocket.accept()

    call_sid = None
    agent = None
    history: list[dict] = []

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "setup":
                call_sid = data.get("callSid", "unknown")
                custom_params = data.get("customParameters") or data.get("custom_parameters") or {}
                agent_id = custom_params.get("agentId", "default")
                agent = agent_manager.get_agent(agent_id) or agent_manager.get_agent("default")

                conversation_store.create_call(
                    call_sid,
                    agent["id"],
                    data.get("from", ""),
                    data.get("to", ""),
                    data.get("direction", "inbound"),
                )
                init_call_log(
                    call_sid,
                    agent_id=agent["id"],
                    from_number=data.get("from", ""),
                    to_number=data.get("to", ""),
                    direction=data.get("direction", "inbound"),
                )
                # The welcome greeting was already spoken by Twilio (it's set
                # on the <ConversationRelay welcomeGreeting=...> attribute) -
                # record it so Claude has it as conversation context too.
                conversation_store.add_message(call_sid, "assistant", agent["welcome_greeting"])
                log_turn(call_sid, role="assistant", content=agent["welcome_greeting"])
                history.append({"role": "assistant", "content": agent["welcome_greeting"]})
                logger.info("[%s] setup complete, agent=%s", call_sid, agent["id"])

            elif msg_type == "prompt":
                if not data.get("last", True):
                    continue  # interim/partial transcript, wait for the final one
                if agent is None:
                    continue  # setup hasn't arrived yet; shouldn't happen

                user_text = data.get("voicePrompt", "").strip()
                if not user_text:
                    continue

                sentiment = await claude_client.classify_sentiment(user_text)
                conversation_store.add_message(
                    call_sid,
                    "user",
                    user_text,
                    sentiment_label=sentiment["sentiment"],
                    sentiment_score=sentiment["score"],
                    emotion=sentiment["emotion"],
                    escalate=sentiment["escalate"],
                )
                log_turn(
                    call_sid,
                    role="user",
                    content=user_text,
                    sentiment=sentiment,
                )
                if sentiment["escalate"]:
                    logger.warning("[%s] ESCALATION FLAG: caller sounds %s", call_sid, sentiment['emotion'])

                claude_facing_text = user_text
                if sentiment["sentiment"] == "negative" or sentiment["escalate"]:
                    claude_facing_text = (
                        f"[Caller tone: {sentiment['emotion']}, sentiment={sentiment['sentiment']}] "
                        f"{user_text}"
                    )
                history.append({"role": "user", "content": claude_facing_text})

                system_prompt = agent_manager.build_system_prompt(agent)
                full_reply = ""
                async for chunk in claude_client.generate_reply_stream(
                    system_prompt, history, agent["model"], agent["effort"]
                ):
                    full_reply += chunk
                    await websocket.send_json({"type": "text", "token": chunk, "last": False})

                await websocket.send_json({"type": "text", "token": "", "last": True})

                history.append({"role": "assistant", "content": full_reply})
                conversation_store.add_message(call_sid, "assistant", full_reply)
                log_turn(call_sid, role="assistant", content=full_reply)

            elif msg_type == "interrupt":
                logger.info("[%s] caller interrupted: %s", call_sid, data.get('utteranceUntilInterrupt', ''))

            elif msg_type == "dtmf":
                logger.info("[%s] dtmf digit: %s", call_sid, data.get('digit'))

    except WebSocketDisconnect:
        logger.info("[%s] client disconnected", call_sid)
    except RuntimeError as e:
        # Most commonly: ANTHROPIC_API_KEY isn't set. Tell Twilio (and speak
        # it to the caller) instead of just dropping the socket silently.
        logger.error("[%s] error: %s", call_sid, e)
        try:
            await websocket.send_json(
                {"type": "text", "token": "Sorry, the assistant is not configured correctly.", "last": True}
            )
            await websocket.send_json({"type": "end"})
        except Exception:
            pass
    finally:
        if call_sid:
            conversation_store.end_call(call_sid)
            finalize_call_log(call_sid, status="completed")
