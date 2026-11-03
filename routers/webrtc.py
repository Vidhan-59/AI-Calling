"""WebRTC entry point for browser calls - see
services/voice_pipeline/webrtc_pipeline.py for why it exists alongside
Twilio (phone calls keep using Twilio Media Streams).

  GET   /webrtc         a minimal test page: pick a scenario or agent task, talk
  POST  /webrtc/offer   SDP offer in, SDP answer out; starts the call
  PATCH /webrtc/offer   trickled ICE candidates

Behind the admin key: a browser call creates call rows and spends Sarvam
credits exactly like a phone call does.
"""

import os
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from services import scenario_engine, scenario_store
from services.auth import require_admin_key

router = APIRouter(tags=["webrtc"])

_PAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "webrtc_call.html")


class WebRTCCallOptions(BaseModel):
    mode: Literal["scenario", "agent"] = "scenario"
    scenario_id: str = "shift_confirm"
    language: str = "hi"
    facts: dict[str, str] = {}
    task_id: str | None = None


class WebRTCOffer(BaseModel):
    sdp: str
    type: str
    pc_id: str | None = None
    restart_pc: bool | None = None
    call: WebRTCCallOptions = WebRTCCallOptions()


class WebRTCCandidates(BaseModel):
    pc_id: str
    candidates: list[dict]


@router.get("/webrtc", include_in_schema=False)
async def webrtc_page():
    return FileResponse(_PAGE)


@router.post("/webrtc/offer", dependencies=[Depends(require_admin_key)])
async def webrtc_offer(offer: WebRTCOffer):
    from pipecat.transports.smallwebrtc.request_handler import SmallWebRTCRequest

    from services.voice_pipeline import webrtc_pipeline

    opts = offer.call
    ref = ""
    if offer.pc_id is None:  # a new call, not a renegotiation - validate before connecting
        if opts.mode == "agent":
            if not opts.task_id:
                raise HTTPException(status_code=400, detail="mode=agent needs call.task_id")
        else:
            if not scenario_engine.get_language_config(opts.scenario_id, opts.language):
                raise HTTPException(status_code=404, detail=f"unknown scenario/language {opts.scenario_id}/{opts.language}")
            missing = scenario_engine.missing_facts(opts.scenario_id, opts.facts)
            if missing:
                raise HTTPException(status_code=400, detail=f"missing facts: {', '.join(missing)}")
            ref = f"webrtc{uuid.uuid4().hex[:10]}"
            scenario_store.create(
                call_sid=f"pending:{ref}", scenario_id=opts.scenario_id, language=opts.language,
                to_number="webrtc", facts=opts.facts, attempt=1,
            )

    started: dict = {}

    async def on_connection(connection):
        if opts.mode == "agent":
            started["call_id"] = await webrtc_pipeline.start_agent_call(connection, task_id=opts.task_id)
        else:
            started["call_id"] = await webrtc_pipeline.start_scenario_call(
                connection, scenario_id=opts.scenario_id, language=opts.language, ref=ref,
            )

    answer = await webrtc_pipeline.request_handler.handle_web_request(
        SmallWebRTCRequest(sdp=offer.sdp, type=offer.type, pc_id=offer.pc_id, restart_pc=offer.restart_pc),
        on_connection,
    )
    if offer.pc_id is None and "call_id" not in started:
        raise HTTPException(status_code=500, detail="call failed to start - see server log")
    return {**(answer or {}), **started}


@router.patch("/webrtc/offer", dependencies=[Depends(require_admin_key)])
async def webrtc_candidates(body: WebRTCCandidates):
    from pipecat.transports.smallwebrtc.request_handler import IceCandidate, SmallWebRTCPatchRequest

    from services.voice_pipeline import webrtc_pipeline

    await webrtc_pipeline.request_handler.handle_patch_request(
        SmallWebRTCPatchRequest(
            pc_id=body.pc_id,
            candidates=[
                IceCandidate(candidate=c.get("candidate", ""), sdp_mid=c.get("sdpMid") or c.get("sdp_mid", ""),
                             sdp_mline_index=c.get("sdpMLineIndex") or c.get("sdp_mline_index") or 0)
                for c in body.candidates
            ],
        )
    )
    return {"ok": True}
